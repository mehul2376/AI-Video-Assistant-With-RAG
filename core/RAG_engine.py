from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm_utils import get_llm, invoke_with_retry
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def format_docs(docs):
    if not docs:
        return "No relevant context found in meeting transcript."
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question 
based only on the meeting transcript context provided below.
If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."
Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question 
based only on the meeting transcript context provided below.
If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."
Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    safe_q = question.encode("ascii", errors="replace").decode("ascii")
    print(f"Question: {safe_q}")
    try:
        answer = invoke_with_retry(rag_chain, question)
        safe_a = answer.encode("ascii", errors="replace").decode("ascii")
        print(f"Answer: {safe_a}")
        return answer
    except Exception as e:
        safe_err = str(e).encode("ascii", errors="replace").decode("ascii")
        print(f"Error invoking RAG chain: {safe_err}")
        return f"Sorry, an error occurred while generating the response: {e}"

