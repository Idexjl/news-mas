from src.common.schemas import HeatScorerInput, HeatScorerOutput
from src.common.prompt_loader import make_run_config


async def run_agent(inp: HeatScorerInput) -> HeatScorerOutput:
    # LangSmith tracing pattern for Ollama agents (phase1_judge and relevance_gate follow
    # the same pattern — swap "heat_scorer" for the agent name).
    #
    #   from langchain_ollama import ChatOllama
    #   llm = ChatOllama(model=os.getenv("MODEL_OVERRIDE") or "gemma4")
    #   run_cfg = make_run_config("heat_scorer", run_id=inp.run_id)
    #   response = await llm.ainvoke(messages, config=run_cfg)
    #
    # config= is the standard LangChain mechanism: it propagates run_name, tags, and
    # metadata to every LangSmith trace without touching the model call signature.
    _ = make_run_config("heat_scorer", run_id=inp.run_id)
    raise NotImplementedError("heat_scorer LLM logic not yet implemented")
