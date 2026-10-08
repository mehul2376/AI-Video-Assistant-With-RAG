import os
import sys
import time
import re
from langchain_groq import ChatGroq
from groq import RateLimitError

# Ensure Windows stdout/stderr never crash on unencodable Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def get_llm():
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        model=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_retries=5,
    )


def invoke_with_retry(chain_or_llm, input_data, max_attempts: int = 5):
    """Invoke a LangChain runnable / chain with automatic backoff on Groq 429 rate limit errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return chain_or_llm.invoke(input_data)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit_exceeded" in err_msg or isinstance(e, RateLimitError):
                if attempt == max_attempts:
                    raise e
                match = re.search(r"try again in ([\d\.]+)s", err_msg, re.IGNORECASE)
                wait_time = float(match.group(1)) + 1.0 if match else (attempt * 6.0)
                print(f"[WARNING] Groq rate limit reached (429). Retrying in {wait_time:.1f}s (Attempt {attempt}/{max_attempts})...")
                time.sleep(wait_time)
            else:
                raise e

