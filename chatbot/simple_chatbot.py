"""
Simple Chatbot Test
===================

Quick test of the RAG chatbot with predefined questions.
"""

from chatbot import RAGChatbot

def main():
    print("=" * 80)
    print("SIMPLE RAG CHATBOT TEST")
    print("=" * 80)

    # Initialize chatbot
    chatbot = RAGChatbot(collection_name="recursive_token_collection")

    # Test questions
    test_questions = [
        "What are the main objectives of the Digital Government Policies?",
        "What is data governance?",
        "Tell me about digital transformation initiatives",
        "What are the beneficiary centricity policies?",
        "How does the government handle digital services?"
    ]

    print("\n" + "=" * 80)
    print("RUNNING TEST QUESTIONS")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'#' * 80}")
        print(f"TEST QUESTION {i}/{len(test_questions)}")
        print(f"{'#' * 80}")

        chatbot.chat(question, n_results=3, show_context=False)

        input("\n[Press Enter to continue to next question...]")

    print("\n" + "=" * 80)
    print("ALL TEST QUESTIONS COMPLETED")
    print("=" * 80)
    print("\nTo use interactive mode, run: python chatbot.py")

if __name__ == "__main__":
    main()
