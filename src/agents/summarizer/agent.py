from src.common.schemas import SummarizerInput, SummarizerOutput
from src.common.prompt_loader import make_run_config


async def run_agent(inp: SummarizerInput) -> SummarizerOutput:
    # LangSmith tracing pattern for Anthropic agents (reviewer follows the same pattern).
    # langchain-anthropic auto-traces to LangSmith when LANGCHAIN_TRACING_V2=true;
    # passing config= tags each run with prompt version, agent, run_id, and topic_id.
    #
    #   from langchain_anthropic import ChatAnthropic
    #   import os
    #   model = os.getenv("MODEL_OVERRIDE") or "claude-sonnet-4-6"
    #   llm = ChatAnthropic(model=model)
    #   run_cfg = make_run_config("summarizer", run_id=inp.run_id,
    #                             topic_id=getattr(inp, "topic_id", None))
    #   response = await llm.ainvoke(messages, config=run_cfg)
    _ = make_run_config("summarizer", run_id=inp.run_id)
    raise NotImplementedError("summarizer LLM logic not yet implemented")
