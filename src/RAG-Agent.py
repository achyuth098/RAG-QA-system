import fitz  # PyMuPDF for extracting text from PDFs
from sentence_transformers import SentenceTransformer
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio as gr

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 1: Load and extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Add your PDFs here
# Be sure you have uploaded the PDF: " Attention is All you Need"
pdf_texts = [
    extract_text_from_pdf("C:/Users/nikhi/Downloads/Attention Is All You Need.pdf"),
    # You can add more PDFs here
]

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

# Process all PDF texts into chunks
documents = []
for pdf_text in pdf_texts:
    documents.extend(split_into_chunks(pdf_text))

# Step 3: Create embeddings for documents
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
document_embeddings = embedder.encode(documents)

# Step 4: Build a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Step 5: Load a lightweight LLM (GPT-2 in this example)
model_name = "gpt2"  # GPT-2 for free usage
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Step 6: RAG-based question answering
def rag_qa(query, top_k=2, max_new_tokens=50):
    """Answers a query using RAG."""
    # Step 1: Encode the query and retrieve relevant documents
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    # Step 2: Generate context for the query
    context = " ".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Step 3: Use GPT-2 to generate the answer
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Set pad_token_id to eos_token_id to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,  # Specify only new tokens to generate
        pad_token_id=tokenizer.pad_token_id  # Set the padding token ID
    )

    # Decode the output and clean up
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("Answer:")[-1].strip()  # Extracting only the part after "Answer:"
    return answer

# Example usage
query = "Explain briefly what a Transformer is"
answer = rag_qa(query)
print(f"Question: {query}\nAnswer: {answer}")



def gradio_rag(query):
    return rag_qa(query)

interface = gr.Interface(
    fn=gradio_rag,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="RAG-based Question Answering",
    description="Ask questions about uploaded PDFs using Retrieval-Augmented Generation (RAG)."
)

interface.launch()