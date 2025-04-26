# 🤖 Advanced FAQ Bot with LangGraph

A smart conversational AI bot that answers user questions using semantic search and keyword matching. Built with Python, LangGraph, and SentenceTransformers.

## 🌟 Features
- **Semantic Search** - Uses `all-MiniLM-L6-v2` embeddings for accurate question matching
- **Hybrid Matching** - Combines AI embeddings with keyword fallback
- **Smart Suggestions** - Recommends related questions when unsure
- **Conversation History** - Maintains context using LangGraph state management

## 🛠️ Installation
```bash
git clone https://github.com/your-username/FAQ-Bot.git
cd FAQ-Bot
pip install -r requirements.txt

1.Run the bot:
python faq_bot.py

2.Try these sample questions:

What services do you offer?

How can I contact support?

What are your business hours?

📝 Requirements
Python 3.8+

Libraries in requirements.txt:

langgraph==0.1.0
sentence-transformers==2.2.2
scikit-learn==1.0.2
numpy==1.22.3

⚠️ Troubleshooting
If you get TensorFlow warnings:

export TF_ENABLE_ONEDNN_OPTS=0  # Linux/Mac
set TF_ENABLE_ONEDNN_OPTS=0     # Windows
