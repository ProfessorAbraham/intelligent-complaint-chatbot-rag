
import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load ChromaDB vector store
client = chromadb.Client(Settings(persist_directory="vector_store", anonymized_telemetry=False))
collection = client.get_or_create_collection(name="complaints")

# Load embedding and generation models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)

# Retrieve top-k relevant chunks
def retrieve_chunks(question, k=5):
    embedded_q = embedder.encode(question, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[embedded_q], n_results=k)
    return results["documents"][0]

# Build the prompt for the LLM
def build_prompt(chunks, question):
    context = "\n".join(chunks)
    return f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.

Use only the information from the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""

# Main RAG function
def answer_question(question):
    try:
        chunks = retrieve_chunks(question)
        prompt = build_prompt(chunks, question)
        response = generator(prompt)[0]['generated_text']
        sources = "\n\n".join([f"• {c[:300]}..." for c in chunks])
        return response, sources
    except Exception as e:
        return "Something went wrong.", str(e)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Intelligent Complaint Chatbot (RAG)")
    gr.Markdown("Ask about customer complaints. Example: *Why are people unhappy with Buy Now Pay Later?*")

    question = gr.Textbox(label="Ask your question")
    answer = gr.Textbox(label="AI Answer", lines=5)
    sources = gr.Textbox(label="Sources", lines=10)

    submit = gr.Button("Ask")
    submit.click(fn=answer_question, inputs=[question], outputs=[answer, sources])

    clear = gr.Button("Clear")
    clear.click(fn=lambda: ("", "", ""), inputs=[], outputs=[question, answer, sources])

demo.launch()
