"""
Streamlit Web UI for RAG Chatbot
=================================

A beautiful web interface for the RAG chatbot.

To run:
    streamlit run chatbot_ui.py

To install Streamlit:
    pip install streamlit
"""

try:
    import streamlit as st
    from chatbot import RAGChatbot
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install Streamlit:")
    print("  pip install streamlit")
    exit(1)

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Document Q&A",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG Chatbot - Ask Questions About Your Documents")
st.markdown("""
Ask questions about the Digital Government Policies document.
The chatbot uses **semantic search** with ChromaDB and **Google Gemini** (free) to answer.
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Collection selector
    collection_name = st.selectbox(
        "Select Collection",
        [
            "recursive_token_collection",
            "fixed_token_collection",
            "cluster_semantic_collection",
            "kamradt_semantic_collection",
            "llm_semantic_collection"
        ],
        index=0
    )

    # Number of results
    n_results = st.slider("Number of chunks to retrieve", 1, 10, 3)

    # Show context toggle
    show_context = st.checkbox("Show retrieved context", value=False)

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    - **Vector DB**: ChromaDB
    - **Embeddings**: Gemini 768-dim
    - **LLM**: Gemini 1.5 Flash (FREE)
    """)

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - What are the Digital Government Policies?
    - What is data governance?
    - Tell me about digital transformation
    - What are the main pillars?
    - What is beneficiary centricity?
    """)

# Initialize chatbot in session state
if 'chatbot' not in st.session_state or st.session_state.get('collection_name') != collection_name:
    with st.spinner(f'Initializing chatbot with {collection_name}...'):
        st.session_state.chatbot = RAGChatbot(collection_name=collection_name)
        st.session_state.collection_name = collection_name
    st.success(f"Connected to: {collection_name}")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show context if available
        if "context" in message and message["context"]:
            with st.expander("📄 Retrieved Context"):
                for i, chunk in enumerate(message["context"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve context
            context = st.session_state.chatbot.retrieve_context(prompt, n_results)

            # Generate answer
            answer = st.session_state.chatbot.generate_answer(prompt, context)

            # Display answer
            st.markdown(answer)

            # Show context if enabled
            if show_context:
                with st.expander("📄 Retrieved Context"):
                    for i, chunk in enumerate(context, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        st.markdown("---")

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context if show_context else None
    })

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with ❤️ using Streamlit, ChromaDB, and Google Gemini (FREE) |
    <a href='https://github.com' target='_blank'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)
