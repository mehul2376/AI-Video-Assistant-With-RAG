from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing core modules

from utils.audio_processing import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractore import extract_action_items, extract_key_decision, extract_question
from core.RAG_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")

    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    print(f"Raw transcription (first 300 chars): {transcript[:300]}")

    rag_chain = build_rag_chain(transcript)

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decision(transcript)
    open_questions = extract_question(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }




if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")