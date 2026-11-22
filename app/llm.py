import json
import re

from . import config


def get_llm():
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.LLM_MODEL, temperature=0, max_retries=2)
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=config.LLM_MODEL, temperature=0, max_tokens=4096)


def text_of(message) -> str:
    """Return plain text from a LangChain message (handles block-style content)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of model text (handles ```json fences and chatter around it)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the model reply")
    return json.loads(text[start : end + 1])


def invoke_structured(prompt: str, schema):
    """Return an instance of the pydantic `schema` produced by the LLM.

    Anthropic/OpenAI: native structured output (tool calling).
    Groq: JSON mode + our own parsing/validation, because open models sometimes
    answer in plain text instead of calling the tool, which Groq rejects with a 400.
    """
    llm = get_llm()
    if config.LLM_PROVIDER != "groq":
        result = llm.with_structured_output(schema).invoke(prompt)
        if result is None:
            raise RuntimeError("The LLM returned no structured output.")
        return result

    full = (
        prompt
        + "\n\nRespond with ONLY one JSON object (no markdown, no commentary) that follows "
        + "this JSON schema exactly:\n"
        + json.dumps(schema.model_json_schema())
    )
    json_llm = llm.bind(response_format={"type": "json_object"})
    last_err = None
    for attempt in range(2):
        msg = json_llm.invoke(
            full if attempt == 0
            else full + f"\n\nYour previous reply was invalid ({last_err}). Return valid JSON only."
        )
        try:
            return schema.model_validate(extract_json(text_of(msg)))
        except Exception as e:  # bad JSON or schema mismatch -> one more try
            last_err = str(e)[:300]
    raise RuntimeError(f"The LLM did not return valid JSON: {last_err}")