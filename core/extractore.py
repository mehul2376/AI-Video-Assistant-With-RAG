# Actionable item, decision, question extraction

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm_utils import get_llm, invoke_with_retry


def build_chain(system_prompt: str):
    llm = get_llm()

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No action items found."

    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each item provide:\n"
        "- Task description\n"
        "- Owner (Who is responsible)\n"
        "- Deadline (If mentioned, else write 'not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    # Use first 12,000 chars (~3000 tokens) to keep within TPM rate limits
    return invoke_with_retry(chain, transcript[:12000])


def extract_key_decision(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No key decisions found."

    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )

    return invoke_with_retry(chain, transcript[:12000])


def extract_question(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No open questions found."

    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )

    return invoke_with_retry(chain, transcript[:12000])
