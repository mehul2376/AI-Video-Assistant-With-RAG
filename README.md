# 🎬 AI Video & Meeting Assistant

An end-to-end AI-powered meeting intelligence system built with **LangChain**, **Groq**, **Whisper**, **Sarvam AI**, **Chroma Vector DB**, and **Streamlit**.

Transform YouTube video URLs or local video/audio files into structured summaries, action items, key decisions, open questions, and an interactive RAG chat interface.

---

## ✨ Features

- 🔊 **Multi-Lingual Audio Processing & Transcription**:
  - **English**: High-speed local transcription using `faster-whisper` / OpenAI `whisper`.
  - **Hinglish / Hindi**: Real-time translation and transcription to English using `Sarvam AI`.
- 🧠 **Parallel LLM Pipeline**:
  - Automatic meeting title generation.
  - Bullet-point meeting summarization.
  - Action items extraction with task, owner, and deadline.
  - Key decisions and unresolved open questions extraction.
- 💬 **Interactive RAG Chat**:
  - Contextual Q&A over meeting transcripts using `Chroma` vector store and `LangChain` LCEL chains.
- ⚡ **Rate-Limit Resilient**:
  - Built-in exponential backoff and retry handling (`429 RateLimitError`) for Groq API limits.
- 🎨 **Modern Streamlit Dashboard**:
  - Dark-mode responsive UI with real-time status notifications.

---

## 🛠️ Architecture & Tech Stack

- **Frontend**: Streamlit
- **LLM Orchestration**: LangChain (LCEL) & ChatGroq
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Speech-to-Text**: `faster-whisper` / `openai-whisper` & `Sarvam AI`
- **Audio Processing**: PyDub, `yt-dlp`, FFmpeg

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- [FFmpeg](https://ffmpeg.org/) installed and added to system PATH.

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/mehul2376/AI-Video-Assistant-With-RAG.git
cd "RAG WIth video ASSITENT"

# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r Requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory (refer to `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key_here
WHISPER_MODEL=small
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_STT_MODEL=saaras:v2.5
GROQ_MODEL=openai/gpt-oss-20b
```

### 4. Running the Web App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📜 License

MIT License
