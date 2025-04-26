from typing import Dict, List, Optional, TypedDict, Tuple
from langgraph.graph import END, StateGraph
import re
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Hides most TensorFlow logs
import warnings
warnings.filterwarnings("ignore")           # Silences Python warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore")

# --- FAQ Data (Predefined Q&A Pairs) ---
FAQ_DATA = [
    {
        "question": "What services does your company offer?",
        "answer": "Our company offers web development, mobile app development, UI/UX design, and digital marketing services.",
        "tags": ["services", "offerings", "what we do"]
    },
    {
        "question": "What are your business hours?",
        "answer": "Our business hours are Monday through Friday, 9 AM to 5 PM Eastern Time.",
        "tags": ["hours", "availability", "contact time"]
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact our customer support team via email at support@example.com or by phone at (555) 123-4567.",
        "tags": ["contact", "support", "help"]
    },
    {
        "question": "Do you offer refunds?",
        "answer": "Yes, we offer full refunds within 30 days of purchase if you're not satisfied with our services.",
        "tags": ["refund", "money back", "cancellation"]
    },
    {
        "question": "What is your pricing structure?",
        "answer": "Our pricing is project-based. We offer free consultations to discuss your needs and provide a custom quote.",
        "tags": ["pricing", "cost", "rates"]
    },
    {
        "question": "How long does a project take?",
        "answer": "Small projects take 2-4 weeks, while larger ones may take months. We provide a timeline during consultation.",
        "tags": ["timeline", "duration", "project length"]
    }
]

# --- State Definition (TypedDict) ---
class BotState(TypedDict):
    messages: List[Dict[str, str]]  # Conversation history
    current_question: Optional[str]  # Latest user question
    found_answer: Optional[str]     # Best-matched answer
    confidence: Optional[float]    # Match confidence (0-1)
    suggested_questions: Optional[List[str]]  # Recommended follow-ups

# --- Advanced FAQ Bot Class ---
class AdvancedFAQBot:
    def __init__(self):
        # Load a lightweight sentence transformer model for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast
        
        # Precompute embeddings for all FAQ questions
        self.faq_questions = [item["question"] for item in FAQ_DATA]
        self.faq_embeddings = self.model.encode(self.faq_questions)
        
        # Stop words for keyword extraction
        self.stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'do', 'does',
            'can', 'could', 'would', 'should', 'the', 'a', 'an', 'in', 'on', 'at',
            'to', 'for', 'with', 'by', 'about', 'as', 'of', 'from', 'your', 'our'
        }
        
        # Build keyword index for faster retrieval
        self.keyword_index = self._build_keyword_index(FAQ_DATA)
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()

    # --- Helper Functions ---
    def _build_keyword_index(self, faq_data: List[Dict]) -> Dict[str, List[int]]:
        """Builds an inverted index mapping keywords to FAQ indices."""
        index = defaultdict(list)
        for idx, entry in enumerate(faq_data):
            keywords = self._extract_keywords(entry["question"])
            for keyword in keywords:
                index[keyword].append(idx)
            if "tags" in entry:
                for tag in entry["tags"]:
                    index[tag.lower()].append(idx)
        return index

    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts important keywords from text (removes stopwords)."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in self.stop_words and len(word) > 2]

    def _semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Finds the most semantically similar FAQ questions using embeddings."""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.faq_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

    def _get_suggested_questions(self, keywords: List[str]) -> List[str]:
        """Recommends questions based on keyword matches."""
        suggested_indices = set()
        for keyword in keywords:
            if keyword in self.keyword_index:
                suggested_indices.update(self.keyword_index[keyword])
        return [FAQ_DATA[idx]["question"] for idx in list(suggested_indices)[:3]]

    # --- LangGraph Nodes ---
    def retrieve_answer(self, state: BotState) -> BotState:
        """Finds the best answer using semantic + keyword matching."""
        if not state["messages"]:
            return state

        user_question = state["messages"][-1]["content"]
        
        # Semantic search (embeddings)
        semantic_matches = self._semantic_search(user_question)
        best_idx, best_score = semantic_matches[0]
        
        # Keyword-based fallback if confidence is low
        if best_score < 0.5:
            keywords = self._extract_keywords(user_question)
            suggested_questions = self._get_suggested_questions(keywords)
            
            if not suggested_questions:
                answer = "I couldn't find an answer. Here are some questions I can answer:\n" + \
                         "\n".join(f"- {q}" for q in self.faq_questions[:3])
                return {**state, "found_answer": answer, "confidence": 0.0}
            
            answer = f"Did you mean:\n" + "\n".join(f"- {q}" for q in suggested_questions)
            return {**state, "found_answer": answer, "confidence": 0.3}
        
        answer = FAQ_DATA[best_idx]["answer"]
        return {**state, "found_answer": answer, "confidence": best_score}

    def format_response(self, state: BotState) -> BotState:
        """Formats the bot's response and updates conversation history."""
        if "found_answer" not in state:
            return state
        
        bot_response = {"role": "assistant", "content": state["found_answer"]}
        state["messages"].append(bot_response)
        return state

    # --- Workflow Construction ---
    def _build_workflow(self):
        """Builds the LangGraph state machine."""
        workflow = StateGraph(BotState)
        
        workflow.add_node("retrieve_answer", self.retrieve_answer)
        workflow.add_node("format_response", self.format_response)
        
        workflow.add_edge("retrieve_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        workflow.set_entry_point("retrieve_answer")
        return workflow.compile()

    # --- Main Interaction Loop ---
    def run(self):
        """Runs the FAQ bot in an interactive CLI loop."""
        print("🌟 FAQ Bot: Hello! Ask me anything. Type 'exit' to quit.\n")
        print("Here are some example questions:")
        for i, q in enumerate(self.faq_questions[:3], 1):
            print(f"{i}. {q}")

        state = {
            "messages": [],
            "current_question": None,
            "found_answer": None,
            "confidence": None,
            "suggested_questions": None
        }

        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nFAQ Bot: Goodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                state["messages"].append({"role": "user", "content": user_input})
                state = self.workflow.invoke(state)
                
                print(f"\nFAQ Bot: {state['messages'][-1]['content']}")
                
            except KeyboardInterrupt:
                print("\nFAQ Bot: Session ended.")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}")
                continue

if __name__ == "__main__":
    bot = AdvancedFAQBot()
    bot.run()