"""Print a new Fernet key to stdout.  Add it to .env as FERNET_KEY=<value>."""
from cryptography.fernet import Fernet

print(Fernet.generate_key().decode())
