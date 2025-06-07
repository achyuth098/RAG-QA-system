import fitz  # PyMuPDF for extracting text from PDFs
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
import gradio as gr

# Initialize Hugging Face Inference Client
HUGGINGFACE_API_KEY = "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"  # Write down here with your own Hugging Face API token
client = InferenceClient(model="tiiuae/falcon-7b-instruct", token=HUGGINGFACE_API_KEY)

# Step 1: Load and extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Split text into manageable chunks
def split_into_chunks(text, max_length=500):
    """Splits text into smaller chunks of a specified max length."""
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        if len(" ".join(chunk)) > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Step 3: Build FAISS index for embeddings
def build_faiss_index(documents):
    """Builds a FAISS index for document embeddings."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
    embeddings = embedder.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, documents

# Step 4: Clean repetitive responses
def clean_response(response):
    """Removes duplicate sentences from the response."""
    sentences = response.split(". ")
    unique_sentences = list(dict.fromkeys(sentences))
    return ". ".join(unique_sentences).strip()

# Step 5: RAG-based question answering
def rag_qa(query, index, documents, top_k=2):
    """Answers a query using RAG with Falcon-7B-Instruct."""
    # Step 1: Retrieve relevant documents
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    # Step 2: Generate context for the query
    context = " ".join(retrieved_docs[:2])[:1000]  # Limit to 1000 characters
    prompt = f"Context:\n{context}\n\nUsing the provided context, answer the question:\n{query}\nAnswer:"

    # Step 3: Use Falcon-7B-Instruct to generate the answer
    try:
        response = client.text_generation(prompt, max_new_tokens=300, temperature=0.2)
        answer = response.strip()  # Directly use the string response
    except Exception as e:
        return f"Error generating response: {str(e)}"

    return clean_response(answer)


# Step 6: Gradio interface
def gradio_rag(query):
    """Interface for RAG QA."""
    return rag_qa(query, index, documents)

# Load your PDFs and prepare the system
# Be sure you have uploaded the PDF file "Attention is all you Need"
# and updated its path
pdf_texts = [extract_text_from_pdf("C:/Users/nikhi/Downloads/Attention Is All You Need.pdf")]
documents = []
for pdf_text in pdf_texts:
    documents.extend(split_into_chunks(pdf_text))
index, documents = build_faiss_index(documents)

# Define the Gradio app
interface = gr.Interface(
    fn=gradio_rag,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="RAG-based Question Answering with Falcon-7B-Instruct",
    description="Ask questions based on the provided PDFs using Retrieval-Augmented Generation (RAG)."
)

# Launch the Gradio interface
interface.launch()