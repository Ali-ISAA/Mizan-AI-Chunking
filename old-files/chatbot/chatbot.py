"""
RAG Chatbot with ChromaDB and Gemini
=====================================

A chatbot that answers questions based on your document chunks stored in ChromaDB.
Uses Google Gemini (free) as the LLM and retrieves relevant context from ChromaDB.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime
from rank_bm25 import BM25Okapi
import re
from typing import List, Tuple, Dict

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add parent directory to path to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import google.generativeai as genai
from other_chunkers.text_chunking_methods import get_gemini_embeddings, GEMINI_API_KEYS

# Note: API keys are managed by text_chunking_methods with automatic rotation
print(f"Chatbot using {len(GEMINI_API_KEYS)} API keys with automatic rotation")


class RAGChatbot:
    """
    Retrieval-Augmented Generation Chatbot using ChromaDB and Gemini.
    """

    def __init__(self, collection_name=None, enable_logging=True):
        """
        Initialize the chatbot.

        Parameters:
        -----------
        collection_name : str, optional
            Name of the ChromaDB collection to query. If None, will list available collections.
        enable_logging : bool, optional
            Whether to log questions and retrieved chunks (default: True)
        """
        print("Initializing RAG Chatbot...")

        # Connect to ChromaDB (using environment variables)
        self.client = chromadb.CloudClient(
            api_key=os.getenv('CHROMADB_API_KEY'),
            tenant=os.getenv('CHROMADB_TENANT'),
            database=os.getenv('CHROMADB_DATABASE')
        )
        print("✓ Connected to ChromaDB Cloud")

        # Get collection
        if collection_name is None:
            collection_name = self._select_collection()

        self.collection_name = collection_name
        self.collection = self.client.get_collection(name=collection_name)
        print(f"✓ Connected to collection: {collection_name}")

        # Initialize Gemini LLM (free model) with generation config
        generation_config = {
            'temperature': 0.1,  # Lower temperature for more focused, deterministic answers
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-lite',
            generation_config=generation_config
        )
        print("Gemini LLM initialized (gemini-2.0-flash-lite - FREE)")

        # Initialize conversation history (last 10 messages)
        self.conversation_history = []

        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging()

        # Initialize BM25 index for keyword search
        print("Building BM25 keyword search index...")
        self._initialize_bm25_index()
        print("✓ BM25 index ready for hybrid search")

        print("\nChatbot ready! Ask questions about your documents.\n")

    def _select_collection(self):
        """
        List available collections and let user select one.

        Returns:
        --------
        str
            Selected collection name
        """
        print("\n" + "=" * 80)
        print("AVAILABLE COLLECTIONS")
        print("=" * 80)

        # List all collections
        collections = self.client.list_collections()

        if not collections:
            print("\n⚠️  No collections found in ChromaDB!")
            print("Please run llm_semantic_md_chunker.py first to create a collection.")
            raise ValueError("No collections available")

        print(f"\nFound {len(collections)} collection(s):\n")
        for i, coll in enumerate(collections, 1):
            # Get collection stats
            try:
                c = self.client.get_collection(name=coll.name)
                count = c.count()
                print(f"{i}. {coll.name}")
                print(f"   Documents: {count}")
            except:
                print(f"{i}. {coll.name}")

        print(f"\n{len(collections) + 1}. Enter custom collection name")

        choice = input(f"\nSelect collection (1-{len(collections) + 1}): ").strip()

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(collections):
                return collections[choice_num - 1].name
            elif choice_num == len(collections) + 1:
                custom_name = input("\nEnter collection name: ").strip()
                # Verify it exists
                try:
                    self.client.get_collection(name=custom_name)
                    return custom_name
                except:
                    print("❌ Collection not found!")
                    raise ValueError(f"Collection '{custom_name}' not found")
            else:
                print("❌ Invalid selection!")
                raise ValueError("Invalid selection")
        except ValueError as e:
            if "invalid literal" in str(e):
                print("❌ Invalid input!")
            raise

    def _setup_logging(self):
        """
        Setup logging directory and session file.
        """
        # Create chatlog directory in parent folder
        self.log_dir = Path(__file__).parent.parent / "chatlog"
        self.log_dir.mkdir(exist_ok=True)

        # Create session log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{timestamp}_{self.collection_name}.json"

        # Initialize session log
        self.session_log = {
            "session_start": datetime.now().isoformat(),
            "collection": self.collection_name,
            "interactions": []
        }

        print(f"✓ Logging enabled: {self.session_log_file.name}")

    def _log_interaction(self, query: str, context: list, answer: str, scores: list = None):
        """
        Log a question-answer interaction with retrieved chunks.

        Parameters:
        -----------
        query : str
            User's question
        context : list
            Retrieved document chunks
        answer : str
            Generated answer
        scores : list, optional
            Retrieval scores for each chunk (hybrid or distance scores)
        """
        if not self.enable_logging:
            return

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "chunk_id": i + 1,
                    "content": chunk,
                    "score": scores[i] if scores else None
                }
                for i, chunk in enumerate(context)
            ],
            "num_chunks": len(context)
        }

        self.session_log["interactions"].append(interaction)

        # Save to file after each interaction
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_log, f, ensure_ascii=False, indent=2)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 (handles both English and Arabic).

        Parameters:
        -----------
        text : str
            Text to tokenize

        Returns:
        --------
        List[str]
            List of tokens
        """
        # Remove special characters but keep Arabic and English letters/numbers
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text.lower())
        # Split on whitespace
        tokens = text.split()
        return tokens

    def _initialize_bm25_index(self):
        """
        Build BM25 index from all documents in the collection.
        """
        # Get all documents from ChromaDB
        all_docs = self.collection.get()

        # Store document texts and IDs
        self.all_document_texts = all_docs['documents']
        self.all_document_ids = all_docs['ids']

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in self.all_document_texts]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"  Indexed {len(self.all_document_texts)} documents for keyword search")

    def _bm25_search(self, query: str, n_results: int = 100) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search.

        Parameters:
        -----------
        query : str
            Search query
        n_results : int
            Number of results to return

        Returns:
        --------
        List[Tuple[str, float]]
            List of (document_text, bm25_score) tuples
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top N results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = [(self.all_document_texts[i], scores[i]) for i in top_indices]

        return results

    def retrieve_context(self, query: str, n_results: int = 100, use_hybrid: bool = True):
        """
        Retrieve relevant document chunks using hybrid search (semantic + keyword).

        Parameters:
        -----------
        query : str
            User's question
        n_results : int
            Number of relevant chunks to retrieve (default: 100)
        use_hybrid : bool
            Whether to use hybrid search (default: True)

        Returns:
        --------
        tuple: (List[str], List[float])
            List of relevant document chunks and their combined scores
        """
        if not use_hybrid:
            # Fall back to pure semantic search
            return self._semantic_search_only(query, n_results)

        # 1. Semantic Search (Embeddings)
        query_embedding = get_gemini_embeddings([query])[0]
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2  # Get more candidates for merging
        )

        # Extract semantic search results
        semantic_docs = semantic_results['documents'][0]
        semantic_distances = semantic_results['distances'][0]

        # 2. Keyword Search (BM25)
        bm25_results = self._bm25_search(query, n_results=n_results * 2)

        # 3. Merge and Rerank using Reciprocal Rank Fusion (RRF)
        merged_results = self._merge_results(
            semantic_docs, semantic_distances,
            bm25_results,
            n_results=n_results
        )

        # Extract final documents and scores
        final_docs = [doc for doc, score in merged_results]
        final_scores = [score for doc, score in merged_results]

        return final_docs, final_scores

    def _semantic_search_only(self, query: str, n_results: int = 15):
        """
        Fallback to pure semantic search (original method).

        Parameters:
        -----------
        query : str
            User's question
        n_results : int
            Number of results

        Returns:
        --------
        tuple: (List[str], List[float])
            Documents and distances
        """
        # Embed the query
        query_embedding = get_gemini_embeddings([query])[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Extract documents
        documents = results['documents'][0]
        distances = results['distances'][0]

        # Filter out very dissimilar results (distance > 0.7 - more lenient)
        filtered_docs = []
        filtered_distances = []
        for doc, dist in zip(documents, distances):
            if dist < 0.7:  # More lenient threshold to include more relevant context
                filtered_docs.append(doc)
                filtered_distances.append(dist)

        # Return at least top 3 if filtering removed everything
        if not filtered_docs:
            return documents[:3], distances[:3]

        return filtered_docs, filtered_distances

    def _merge_results(
        self,
        semantic_docs: List[str],
        semantic_distances: List[float],
        bm25_results: List[Tuple[str, float]],
        n_results: int = 15
    ) -> List[Tuple[str, float]]:
        """
        Merge semantic and keyword search results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for each result list
        where k = 60 (constant), rank_i = rank in result list i

        Parameters:
        -----------
        semantic_docs : List[str]
            Documents from semantic search
        semantic_distances : List[float]
            Distances from semantic search (lower = more similar)
        bm25_results : List[Tuple[str, float]]
            Results from BM25 keyword search
        n_results : int
            Number of final results to return

        Returns:
        --------
        List[Tuple[str, float]]
            Merged and reranked results as (document, rrf_score) tuples
        """
        k = 60  # RRF constant

        # Dictionary to accumulate scores
        doc_scores: Dict[str, float] = {}
        doc_has_keyword_match: Dict[str, bool] = {}

        # Find max BM25 score to normalize
        max_bm25_score = max([score for _, score in bm25_results]) if bm25_results else 0

        # Add BM25 keyword results FIRST (rank by BM25 score - higher is better)
        for rank, (doc, bm25_score) in enumerate(bm25_results, start=1):
            # Only include if BM25 score > 0 (has some relevance)
            if bm25_score > 0:
                rrf_score = 1.0 / (k + rank)

                # Normalize BM25 score (0-1 range)
                normalized_bm25 = bm25_score / max_bm25_score if max_bm25_score > 0 else 0

                # Strong keyword matches get MUCH higher weight
                if normalized_bm25 > 0.5:  # Strong keyword match
                    weight = 3.0  # 3x weight for strong keyword matches
                    doc_has_keyword_match[doc] = True
                elif normalized_bm25 > 0.2:  # Moderate keyword match
                    weight = 2.0  # 2x weight for moderate matches
                    doc_has_keyword_match[doc] = True
                else:  # Weak keyword match
                    weight = 1.0
                    doc_has_keyword_match[doc] = False

                # Add boosted keyword score
                doc_scores[doc] = doc_scores.get(doc, 0) + (rrf_score * weight)

        # Add semantic search results (rank by distance - lower is better)
        for rank, (doc, dist) in enumerate(zip(semantic_docs, semantic_distances), start=1):
            # Only include if distance < 0.8 (reasonably similar)
            if dist < 0.8:
                rrf_score = 1.0 / (k + rank)

                # If document already has strong keyword match, boost semantic score too
                if doc_has_keyword_match.get(doc, False):
                    weight = 1.5  # Extra boost for documents with both keyword + semantic match
                else:
                    weight = 1.0  # Normal weight for semantic-only matches

                doc_scores[doc] = doc_scores.get(doc, 0) + (rrf_score * weight)

        # Sort by combined score (highest first)
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N results
        return sorted_results[:n_results]

    def generate_answer(self, query: str, context: list):
        """
        Generate an answer using Gemini LLM with retrieved context and conversation history.

        Parameters:
        -----------
        query : str
            User's question
        context : list
            List of relevant document chunks

        Returns:
        --------
        str
            Generated answer
        """
        # Combine context chunks
        context_text = "\n\n---\n\n".join(context)

        # Format conversation history (last 10 messages)
        history_text = ""
        if self.conversation_history:
            history_text = "\n\nCONVERSATION HISTORY:\n"
            for msg in self.conversation_history[-10:]:  # Last 10 messages
                history_text += f"{msg['role']}: {msg['content']}\n"

        # Create prompt for n8n documentation expert
        prompt = f"""<role>
You are a specialized AI assistant. Your sole mission is to help users by providing accurate and factual information extracted exclusively from this documentation. You are meticulous, factual, and never deviate from your knowledge scope.
</role>

<context>
==================== N8N DOCUMENTATION CONTEXT ====================
{context_text}
==================================================================
{history_text}
</context>

<user_question>
{query}
</user_question>

<instructions>
<goal>
Your primary goal is to provide precise and factual answers to user questions about the n8n automation platform, based **exclusively** on the excerpts from the official documentation provided in the context above.
</goal>

**Mandatory rules:**

1.  **Single source of truth:** Your answer MUST be entirely and solely derived from the information present in the provided documentation.
2.  **Accuracy and implicit citation:** Base your answer as literally as possible on the documentation text. Rephrase for clarity and conciseness, but do not add any information not found there. Act as if the documentation is your only knowledge in the world.
3.  **Do not mention the process:** Never mention your tool or the fact that you are a RAG system in your answer to the user. Respond as an expert who directly consults their documentation.

<output_format>
*   **Clarity:** Provide a clear, concise, and direct answer.
*   **Structuring:** If the context contains steps, lists, or code examples, use Markdown syntax to format them legibly (bullet points, numbered lists, code blocks for code snippets, JSON, etc.).
*   **Tone:** Adopt a professional, helpful, and confident tone, that of a technical n8n expert.
</output_format>
</instructions>

Now provide your expert answer based on the documentation above:"""

        # Generate response
        try:
            response = self.model.generate_content(prompt)
            answer = response.text

            # Update conversation history
            self.conversation_history.append({"role": "User", "content": query})
            self.conversation_history.append({"role": "Assistant", "content": answer})

            # Keep only last 10 messages (5 Q&A pairs)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return answer
        except Exception as e:
            return f"Error generating response: {e}"

    def chat(self, query: str, n_results: int = 100, show_context: bool = False):
        """
        Main chat function - retrieve context and generate answer.

        Parameters:
        -----------
        query : str
            User's question
        n_results : int
            Number of chunks to retrieve (default: 100)
        show_context : bool
            Whether to display retrieved context (default: False)

        Returns:
        --------
        str
            Generated answer
        """
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print(f"{'='*80}\n")

        # Retrieve relevant context
        print(f"Retrieving relevant information from {self.collection_name}...")
        print("  Using hybrid search (semantic + keyword)...")
        context, scores = self.retrieve_context(query, n_results)
        print(f"Retrieved {len(context)} relevant chunk(s)\n")

        # Show context if requested
        if show_context:
            print("=" * 80)
            print("RETRIEVED CONTEXT:")
            print("=" * 80)
            for i, (chunk, score) in enumerate(zip(context, scores), 1):
                print(f"\n[Chunk {i}] (Hybrid Score: {score:.4f})")
                print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                print("-" * 80)
            print()

        # Generate answer
        print("Generating answer with Gemini...")
        answer = self.generate_answer(query, context)

        print("=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)
        print("=" * 80)

        # Log the interaction
        self._log_interaction(query, context, answer, scores)

        return answer

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n✓ Conversation history cleared")

    def interactive_mode(self):
        """
        Start interactive chat mode.
        """
        print("\n" + "=" * 80)
        print("INTERACTIVE CHAT MODE")
        print("=" * 80)
        print("Ask questions about your documents. Type 'quit' or 'exit' to stop.")
        print("Type 'context' to show retrieved context with answers.")
        print("Type 'collection' to switch collections.")
        print("Type 'clear' to clear conversation history.")
        print("=" * 80 + "\n")

        show_context = False

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break

                if user_input.lower() == 'context':
                    show_context = not show_context
                    status = "enabled" if show_context else "disabled"
                    print(f"\nContext display {status}")
                    continue

                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue

                if user_input.lower() == 'collection':
                    try:
                        new_collection = self._select_collection()
                        self.collection = self.client.get_collection(name=new_collection)
                        self.collection_name = new_collection
                        print(f"\n✓ Switched to collection: {new_collection}")
                        # Clear conversation history when switching collections
                        self.clear_history()
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                    continue

                # Process the question
                self.chat(user_input, n_results=15, show_context=show_context)

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """
    Main function to run the chatbot.
    """
    print("=" * 80)
    print("RAG CHATBOT - Ask Questions About Your Documents")
    print("=" * 80)
    print("\nUsing:")
    print("  - Vector DB: ChromaDB Cloud")
    print("  - Embeddings: Google Gemini embedding-001 (768-dim)")
    print("  - LLM: Google Gemini 2.0 Flash-Lite (FREE)")
    print("  - Chunking: LLM Semantic Analysis with Auto-Split")
    print("  - Search: Hybrid (Semantic + Keyword BM25)")
    print("=" * 80 + "\n")

    # Initialize chatbot (will prompt user to select collection)
    try:
        chatbot = RAGChatbot(collection_name=None)
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        return

    # Ask if user wants to run example questions
    print("\n" + "=" * 80)
    print("GETTING STARTED")
    print("=" * 80)
    print("\nOptions:")
    print("1. Start interactive mode (ask your own questions)")
    print("2. Run example questions first, then interactive mode")
    print("3. Exit")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "2":
        print("\n" + "=" * 80)
        print("EXAMPLE QUESTIONS")
        print("=" * 80)
        print("\nNote: These are generic examples. Modify them based on your document content.")

        example_questions = [
            "What is this document about?",
            "What are the main topics covered?",
            "Summarize the key points"
        ]

        for question in example_questions:
            try:
                chatbot.chat(question, n_results=15, show_context=False)
                print("\n")
            except Exception as e:
                print(f"Error: {e}\n")

    if choice in ["1", "2"]:
        # Start interactive mode
        print("\n" + "=" * 80)
        print("Ready for your questions!")
        print("=" * 80)

        chatbot.interactive_mode()
    else:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
