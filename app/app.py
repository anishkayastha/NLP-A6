# app.py - Contextual Retrieval RAG Chatbot for Chapter 8: Transformers
# Built with Chainlit

import os
import json
import chainlit as cl
from openai import OpenAI
import faiss
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Paths (relative to app/ directory)
BASE_DIR = Path(__file__).parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vectorstore" / "contextual_retrieval" / "index.faiss"
CHUNKS_PATH = BASE_DIR / "data" / "chapter8_contextualized_chunks.json"

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
TOP_K = 3
TEMPERATURE = 0
MAX_TOKENS = 500

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_vectorstore():
    """Load FAISS index and contextualized chunks"""
    try:
        # Load FAISS index
        index = faiss.read_index(str(VECTOR_STORE_PATH))

        # Load chunks
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f'✓ Loaded FAISS index with {index.ntotal} vectors')
        print(f'✓ Loaded {len(chunks)} contextualized chunks')

        return index, chunks

    except Exception as e:
        print(f'❌ Error loading vector store: {e}')
        raise


def get_embedding(text: str):
    """Get embedding from OpenAI text-embedding-3-small"""
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    except Exception as e:
        print(f'❌ Error getting embedding: {e}')
        raise


def retrieve_context(query: str, index, chunks, k: int = TOP_K):
    """Retrieve top-k relevant contextualized chunks"""
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')

        # Search in FAISS index
        distances, indices = index.search(query_vector, k)

        # Format results
        retrieved = []
        for idx, dist in zip(indices[0], distances[0]):
            retrieved.append({
                'chunk': chunks[idx]['text'],
                'page': chunks[idx]['page'],
                'chunk_id': chunks[idx]['chunk_id'],
                'similarity': 1 / (1 + float(dist))  # Convert L2 distance to similarity
            })

        return retrieved

    except Exception as e:
        print(f'❌ Error retrieving context: {e}')
        raise


def generate_answer(query: str, context_chunks: list):
    """Generate answer using GPT-4o-mini"""
    try:
        # Format context from retrieved chunks
        context = '\n\n'.join([
            f"[Page {chunk['page']}]: {chunk['chunk']}"
            for chunk in context_chunks
        ])

        # Create prompt
        prompt = f"""You are a helpful assistant answering questions about Chapter 8: Transformers from the Stanford NLP textbook.

Use only the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f'❌ Error generating answer: {e}')
        raise


# ============================================================================
# CHAINLIT EVENT HANDLERS
# ============================================================================

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    try:
        # Load vector store and chunks
        index, chunks = load_vectorstore()

        # Store in session
        cl.user_session.set("index", index)
        cl.user_session.set("chunks", chunks)

        # Welcome message
        welcome_message = """# Transformers Q&A Bot

Welcome! I can answer questions about **Chapter 8: Transformers** from the Stanford NLP textbook.

This chatbot uses **Contextual Retrieval**, where chunks are enriched with contextual information before embedding for improved relevance.

## Example Questions:
- What is self-attention and how does it work?
- Explain the role of Query, Key, and Value matrices
- How does multi-head attention differ from single-head attention?
- What is positional encoding and why is it important?
- Describe the transformer block architecture
- What are scaling laws in language models?
- How does the KV cache improve inference speed?
- What are induction heads in transformers?

**Ask me anything about transformers!**
"""

        await cl.Message(content=welcome_message).send()

    except Exception as e:
        error_message = f"""❌ **Error initializing chatbot:**

{str(e)}

Please ensure:
1. OPENAI_API_KEY is set in your environment
2. Vector store exists at: `{VECTOR_STORE_PATH}`
3. Chunks file exists at: `{CHUNKS_PATH}`

Run the notebooks 01-03 first to generate the required data.
"""
        await cl.Message(content=error_message).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    # Get vector store from session
    index = cl.user_session.get("index")
    chunks = cl.user_session.get("chunks")

    # Check if vector store is loaded
    if index is None or chunks is None:
        await cl.Message(
            content="❌ Error: Vector store not loaded. Please restart the chat."
        ).send()
        return

    # Show loading message
    loading_msg = cl.Message(content="🔍 Searching through Chapter 8...")
    await loading_msg.send()

    try:
        # Retrieve relevant context
        retrieved = retrieve_context(message.content, index, chunks, k=TOP_K)

        # Update loading message
        loading_msg.content = "💭 Generating answer..."
        await loading_msg.update()

        # Generate answer
        answer = generate_answer(message.content, retrieved)

        # Remove loading message
        await loading_msg.remove()

        # Send answer
        await cl.Message(content=f"**Answer:**\n\n{answer}").send()

        # Format sources
        sources_content = "## 📚 Sources Used:\n\n"

        for i, chunk in enumerate(retrieved):
            # Truncate chunk for display (show first 300 chars)
            chunk_preview = chunk['chunk'][:300]
            if len(chunk['chunk']) > 300:
                chunk_preview += "..."

            sources_content += f"""**Source {i+1}** (Page {chunk['page']}, Similarity: {chunk['similarity']:.3f})

```
{chunk_preview}
```

---

"""

        # Send sources as expandable element
        await cl.Message(
            content=sources_content
        ).send()

    except Exception as e:
        # Remove loading message
        await loading_msg.remove()

        # Send error message
        error_content = f"""❌ **Error processing your question:**

{str(e)}

Please try again or rephrase your question.
"""
        await cl.Message(content=error_content).send()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Verify environment
    if not OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("Set it using: export OPENAI_API_KEY='your-key-here'")

    # Print configuration
    print("\n" + "="*80)
    print("CHAINLIT RAG CHATBOT - CHAPTER 8: TRANSFORMERS")
    print("="*80)
    print(f"✓ Vector store path: {VECTOR_STORE_PATH}")
    print(f"✓ Chunks path: {CHUNKS_PATH}")
    print(f"✓ Embedding model: {EMBEDDING_MODEL}")
    print(f"✓ Generation model: {GENERATION_MODEL}")
    print(f"✓ Top-K retrieval: {TOP_K}")
    print(f"✓ Temperature: {TEMPERATURE}")
    print(f"✓ Max tokens: {MAX_TOKENS}")
    print("="*80)
    print("\nStarting Chainlit app...")
    print("Open your browser and navigate to the URL shown below.\n")
