import html
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables before importing core modules
load_dotenv()

from utils.audio_processing import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractore import (
    extract_action_items,
    extract_key_decision,
    extract_question,
)
from core.RAG_engine import build_rag_chain, ask_question


# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── HTML helper (THE FIX) ──────────────────────────────────────────────────────
def H(s: str) -> str:
    """Flatten HTML into one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def show(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* Bordered containers act as cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 12px !important;
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);   border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);   border: 1px solid rgba(16,185,129,0.3); }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    word-break: break-word;
}

hr { border-top: 1px solid var(--border) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ──────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
}.items():
    st.session_state.setdefault(key, default)


def card(title: str, body: str):
    """Card with a title; body is rendered as normal Markdown (safe)."""
    with st.container(border=True):
        show(f'<div class="card-title">{title}</div>')
        st.markdown(body)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    show('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>')
    show('<div class="hero-sub">Meeting Intelligence</div>')

    st.markdown("---")
    show('<span class="badge badge-purple">Input</span>')

    source = st.text_input(
        "YouTube URL or File Path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("⚡  Analyse", use_container_width=True)


# ─── Main Area ──────────────────────────────────────────────────────────────────
show('<div class="hero-title">AI Video Assistant</div>')
show('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>')
st.markdown("---")


# ─── Run Pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.result = None
        st.session_state.chat_history = []

        try:
            with st.status("⚙️ Pipeline running…", expanded=True) as status:
                st.write("🔊 Audio processing")
                chunks = process_input(source)

                st.write("📝 Transcribing audio")
                transcript = transcribe_all(chunks, language)

                st.write("📌 Building vector store for chat")
                rag_chain = build_rag_chain(transcript)

                st.write("🏷️ Generating session title…")
                title = generate_title(transcript)

                st.write("📋 Generating summary…")
                summary = summarize(transcript)

                st.write("✅ Extracting action items…")
                action_items = extract_action_items(transcript)

                st.write("🔑 Extracting key decisions…")
                key_decisions = extract_key_decision(transcript)

                st.write("❓ Extracting open questions…")
                open_questions = extract_question(transcript)

                st.session_state.result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": key_decisions,
                    "open_questions": open_questions,
                    "rag_chain": rag_chain,
                }

                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"❌ Error: {e}")




# ─── Results ────────────────────────────────────────────────────────────────────
r = st.session_state.result

if r:
    # Title banner
    with st.container(border=True):
        show('<div class="card-title">📌 Session Title</div>')
        show(
            f"""<div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;">
            {html.escape(str(r['title']))}</div>"""
        )

    # Top row
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        card("📋 Summary", str(r["summary"]))

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            safe = html.escape(str(r["transcript"])).replace("\n", "<br>")
            show(f'<div class="transcript-box">{safe}</div>')

    # Second row
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        card("✅ Action Items", str(r["action_items"]))
    with c2:
        card("🔑 Key Decisions", str(r["key_decisions"]))
    with c3:
        card("❓ Open Questions", str(r["open_questions"]))

    st.markdown("---")

    # ── Chat ──
    show('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;">💬 Chat with your Meeting</div>')

    if not st.session_state.chat_history:
        st.caption("Ask anything about your meeting transcript.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("What were the main decisions made?"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─── Empty State ────────────────────────────────────────────────────────────────
else:
    show(
        """
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;text-align:center;">
            <div style="font-size:4rem;margin-bottom:1rem;">🎬</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem;">
                Ready to Analyse
            </div>
            <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7;">
                Paste a YouTube URL or local file path in the sidebar, choose your language, and hit
                <strong>Analyse</strong> to get started.
            </div>
            <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;">
                <span class="badge badge-purple">Transcription</span>
                <span class="badge badge-cyan">Summarisation</span>
                <span class="badge badge-green">RAG Chat</span>
            </div>
        </div>
        """
    )