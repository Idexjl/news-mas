#!/usr/bin/env python3
"""
Wire Microsoft Entra ID application scopes and API permissions for news-mas agents.

Prerequisites:
    pip install requests
    az login   (account must have Application Administrator or Global Administrator role)

Usage:
    python scripts/setup_entra_scopes.py [--dry-run]

Reads:
    entra-app-ids.json    name -> clientId for every app registration
    entra-secrets.json    per-app tenantId / clientId / clientSecret (tenantId used for display only)

Writes:
    entra-scope-map.json  name -> { objectId, clientId, scopeId }  (gitignored)

Three phases:
    1. Add an 'Invoke' oauth2PermissionScope to every app registration (idempotent).
    2. Wire requiredResourceAccess according to the call-chain table below.
    3. Grant admin consent for each app via `az ad app permission admin-consent`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import requests

# ── file paths ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_IDS_FILE = REPO_ROOT / "entra-app-ids.json"
SECRETS_FILE = REPO_ROOT / "entra-secrets.json"
SCOPE_MAP_FILE = REPO_ROOT / "entra-scope-map.json"
GITIGNORE_FILE = REPO_ROOT / ".gitignore"

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# ── call-chain topology ───────────────────────────────────────────────────────
# gateway calls all 8 pipeline agents; reviewer retries through summarizer.
# Every other agent makes no outbound agent calls.
CALL_CHAIN: dict[str, list[str]] = {
    "news-mas-gateway": [
        "news-mas-filter-agent",
        "news-mas-heat-scorer",
        "news-mas-phase1-judge",
        "news-mas-relevance-gate",
        "news-mas-reviewer",
        "news-mas-search-worker",
        "news-mas-selector",
        "news-mas-summarizer",
    ],
    "news-mas-reviewer": [
        "news-mas-summarizer",
    ],
}


# ── Graph API helpers ─────────────────────────────────────────────────────────

def _az_graph_token() -> str:
    """Return a bearer token for Graph API acquired via the az CLI."""
    # On Windows az is a .cmd wrapper; shell=True (with a string command) is
    # required to invoke it correctly. shutil.which resolves the full path so
    # the quoted invocation works even when PATH contains spaces.
    az_cmd = shutil.which("az") or "az"
    result = subprocess.run(
        f'"{az_cmd}" account get-access-token --resource https://graph.microsoft.com',
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        print(f"ERROR: az account get-access-token failed:\n{result.stderr.strip()}", file=sys.stderr)
        print("Hint: run `az login` first, then retry.", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)["accessToken"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _graph_get(token: str, path: str) -> Any:
    resp = requests.get(f"{GRAPH_BASE}{path}", headers=_headers(token), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"GET {path} → {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def _graph_patch(token: str, path: str, body: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        print(f"    [dry-run] PATCH {GRAPH_BASE}{path}")
        print(f"    body: {json.dumps(body, indent=6)}")
        return
    resp = requests.patch(
        f"{GRAPH_BASE}{path}",
        headers=_headers(token),
        json=body,
        timeout=30,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"PATCH {path} → {resp.status_code}: {resp.text[:400]}")


def _get_application(token: str, client_id: str) -> dict[str, Any]:
    """Return the Graph application object for a given client ID (appId)."""
    data = _graph_get(token, f"/applications?$filter=appId eq '{client_id}'")
    values = data.get("value", [])
    if not values:
        raise RuntimeError(f"No application found for appId={client_id!r}. "
                           "Check that entra-app-ids.json is correct and the account has read access.")
    return values[0]


# ── Phase 1: Invoke scope ─────────────────────────────────────────────────────

def _ensure_invoke_scope(
    token: str,
    name: str,
    client_id: str,
    app_obj: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    """
    Add an 'Invoke' oauth2PermissionScope to the app if absent.
    Returns the scope UUID (existing or freshly generated).
    """
    object_id: str = app_obj["id"]
    existing_scopes: list[dict] = app_obj.get("api", {}).get("oauth2PermissionScopes", [])

    existing = next((s for s in existing_scopes if s.get("value") == "Invoke"), None)
    if existing:
        print(f"    [exists] Invoke scope {existing['id']}")
        return existing["id"]

    scope_id = str(uuid.uuid4())
    new_scope: dict[str, Any] = {
        "adminConsentDescription": f"Allow calling agent to invoke {name}",
        "adminConsentDisplayName": f"Invoke {name}",
        "id": scope_id,
        "isEnabled": True,
        "type": "Admin",
        "value": "Invoke",
    }
    print(f"    adding Invoke scope {scope_id}")
    _graph_patch(
        token,
        f"/applications/{object_id}",
        {"api": {"oauth2PermissionScopes": existing_scopes + [new_scope]}},
        dry_run=dry_run,
    )
    return scope_id


# ── Phase 2: requiredResourceAccess ──────────────────────────────────────────

def _wire_caller_permissions(
    token: str,
    caller_name: str,
    caller_client_id: str,
    targets: list[str],
    app_ids: dict[str, str],
    scope_map: dict[str, dict[str, str]],
    *,
    dry_run: bool,
) -> None:
    """
    Add each target's Invoke scope to the caller's requiredResourceAccess.
    Reads the current state fresh, merges new entries, then PATCHes once.
    """
    caller_app = _get_application(token, caller_client_id)
    caller_object_id: str = caller_app["id"]

    # Deep-copy the existing RRA so we can mutate safely.
    rra: list[dict[str, Any]] = [
        {
            "resourceAppId": e["resourceAppId"],
            "resourceAccess": list(e.get("resourceAccess", [])),
        }
        for e in caller_app.get("requiredResourceAccess", [])
    ]

    added: list[str] = []

    for target_name in targets:
        if target_name not in app_ids:
            print(f"    WARNING: {target_name!r} not in entra-app-ids.json — skipping")
            continue
        target_client_id = app_ids[target_name]
        target_scope_id = scope_map[target_name]["scopeId"]

        entry = next((e for e in rra if e["resourceAppId"] == target_client_id), None)
        if entry is not None:
            if any(ra["id"] == target_scope_id for ra in entry["resourceAccess"]):
                print(f"    [exists] → {target_name}")
                continue
            entry["resourceAccess"].append({"id": target_scope_id, "type": "Scope"})
        else:
            rra.append({
                "resourceAppId": target_client_id,
                "resourceAccess": [{"id": target_scope_id, "type": "Scope"}],
            })
        added.append(target_name)
        print(f"    + {target_name}")

    if added:
        _graph_patch(
            token,
            f"/applications/{caller_object_id}",
            {"requiredResourceAccess": rra},
            dry_run=dry_run,
        )
    else:
        print(f"    [no changes needed]")


# ── Phase 3: admin consent ────────────────────────────────────────────────────

def _grant_admin_consent(client_id: str, name: str, *, dry_run: bool) -> None:
    az_cmd = shutil.which("az") or "az"
    cmd = f'"{az_cmd}" ad app permission admin-consent --id {client_id}'
    if dry_run:
        print(f"    [dry-run] {cmd}")
        return
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        # admin-consent can be flaky for apps with no delegated permissions — warn, don't abort.
        print(f"    WARNING ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    else:
        print(f"    OK")


# ── .gitignore guard ──────────────────────────────────────────────────────────

def _ensure_gitignore(entry: str) -> None:
    if not GITIGNORE_FILE.exists():
        return
    contents = GITIGNORE_FILE.read_text(encoding="utf-8")
    if entry in contents.splitlines():
        return
    with GITIGNORE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{entry}\n")
    print(f"Added {entry!r} to .gitignore")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wire Entra ID scopes and API permissions for news-mas agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Graph API calls without executing them. No changes are made.",
    )
    args = parser.parse_args()
    dry_run: bool = args.dry_run

    if dry_run:
        print("=" * 60)
        print("DRY RUN — no changes will be made to Entra ID")
        print("=" * 60)
        print()

    # ── load input files ──────────────────────────────────────────────────────
    if not APP_IDS_FILE.exists():
        print(f"ERROR: {APP_IDS_FILE} not found.", file=sys.stderr)
        sys.exit(1)
    app_ids: dict[str, str] = json.loads(APP_IDS_FILE.read_text(encoding="utf-8"))

    tenant_id: str | None = None
    if SECRETS_FILE.exists():
        secrets: dict = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
        first_secret = next(iter(secrets.values()), {})
        tenant_id = first_secret.get("tenantId")
    else:
        print(f"WARNING: {SECRETS_FILE} not found — skipping tenant validation", file=sys.stderr)

    # Ensure output file is gitignored before writing.
    _ensure_gitignore("entra-scope-map.json")

    # ── acquire Graph token ───────────────────────────────────────────────────
    print("Acquiring Graph API token via az CLI...")
    token = _az_graph_token()
    print("Token acquired.\n")

    # ── Phase 1: add Invoke scope to every app ────────────────────────────────
    print("=" * 60)
    print("Phase 1 — Add Invoke oauth2PermissionScope")
    print("=" * 60)

    scope_map: dict[str, dict[str, str]] = {}

    for name, client_id in app_ids.items():
        print(f"\n  {name}")
        print(f"    clientId: {client_id}")
        app_obj = _get_application(token, client_id)
        object_id: str = app_obj["id"]
        print(f"    objectId: {object_id}")
        scope_id = _ensure_invoke_scope(token, name, client_id, app_obj, dry_run=dry_run)
        scope_map[name] = {
            "objectId": object_id,
            "clientId": client_id,
            "scopeId": scope_id,
        }

    if not dry_run:
        SCOPE_MAP_FILE.write_text(json.dumps(scope_map, indent=2), encoding="utf-8")
        print(f"\nScope map saved → {SCOPE_MAP_FILE.name}")

    # ── Phase 2: wire requiredResourceAccess ──────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 2 — Wire requiredResourceAccess (call-chain)")
    print("=" * 60)

    for caller_name, targets in CALL_CHAIN.items():
        if caller_name not in app_ids:
            print(f"\n  WARNING: caller {caller_name!r} not in entra-app-ids.json — skipping")
            continue
        print(f"\n  {caller_name}  →  {', '.join(targets)}")
        _wire_caller_permissions(
            token,
            caller_name,
            app_ids[caller_name],
            targets,
            app_ids,
            scope_map,
            dry_run=dry_run,
        )

    # ── Phase 3: admin consent ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 3 — Grant admin consent")
    print("=" * 60)

    for name, client_id in app_ids.items():
        print(f"\n  {name}")
        _grant_admin_consent(client_id, name, dry_run=dry_run)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if tenant_id:
        print(f"  Tenant ID : {tenant_id}")
    print(f"  Apps      : {len(scope_map)}")
    print()
    for name, info in scope_map.items():
        print(f"  {name}")
        print(f"    objectId : {info['objectId']}")
        print(f"    scopeId  : {info['scopeId']}")
    print()
    print("  Call-chain permissions wired:")
    for caller, targets in CALL_CHAIN.items():
        for t in targets:
            print(f"    {caller}  →  {t}")
    if not dry_run:
        print()
        print(f"  Scope map written to: {SCOPE_MAP_FILE}")
    print()
    if dry_run:
        print("  (dry-run: no changes were made)")


if __name__ == "__main__":
    main()
