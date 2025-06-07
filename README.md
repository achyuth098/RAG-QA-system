# RAG QA System

A **Retrieval-Augmented Generation** question-answering pipeline that combines fast vector-based retrieval with a large language model to give accurate, context-aware answers over your own document collection.

---

## 🔍 Description

This project implements a RAG-based QA system end-to-end:

## **1. GPT2-based RAG Agent**


This first general-purpose generative model is based on **GPT-2**, a small LLM with 117M to 1.5B parameters. The model is used locally, and mainly based on the ***transformers*** library. Can run on CPU or GPU. Employing this lightweight LLM may lead to less accurate responses and an increased likelihood of hallucination.  

Its basic architecture uses PyMuPDF, SentenceTransformer, FAISS, and GPT-2. No pre-built framework like LangChain is used, but the code integrates the pieces manually.

**Vector store:** FAISS (Facebook AI Similarity Search) Index (faiss.IndexFlatL2) is used to store embeddings and retrieve the top-k similar documents.

**LLM used:** GPT-2 (GPT2LMHeadModel) is used as the language model for generating responses based on the prompt.

**Embeddings producer:** SentenceTransformer (all-MiniLM-L6-v2) generates embeddings for document chunks and the query.

**PDF (external documents) manager:** PyMuPDF (fitz) reads the content of PDF files, page by page.

## **2. Falcon-based RAG Agent**

This RAG app combines document retrieval and advanced language generation for effective question-answering. It extracts text from PDFs using **PyMuPDF**, splits the text into manageable chunks for processing, and generates embeddings with ***SentenceTransformer** for semantic similarity.

The ***FAISS index*** ensures efficient retrieval of relevant document chunks, which are then passed to ***Falcon-7B-Instruct via Hugging Face's Inference Client*** for generating accurate and context-aware responses.

A Gradio interface provides an interactive and user-friendly platform, allowing users to query documents without technical expertise.

Here the most relevant differences in front of model 1:

- Thos second  model integrates Hugging Face's Inference Client, enabling seamless use of a powerful hosted LLM like Falcon-7B-Instruct, which surpasses GPT-2 in generating complex, instruction-based responses.  

- Uses advanced generation parameters such as `temperature`, `top_k`, and `max_new_tokens`, improving  response control and quality, while also implementing cleaning functions to eliminate repetitive outputs.

- This second app processes context efficiently, limiting it to 1000 characters to ensure compliance with model token limits without compromising relevancy.

- By leveraging the Falcon model's state-of-the-art instruction-following capabilities, the app delivers richer, contextually accurate, and nuanced answers compared to the lightweight GPT-2 approach.  

In this model, ***Hugging Face*** provides the Falcon-7B-Instruct model, a powerful, instruction-tuned large language model (LLM). Additionally, The ***InferenceClient*** from the Hugging Face Hub is used to send prompts to the hosted model and retrieve responses.

The Hugging Face API token (HUGGINGFACE_API_KEY) authenticates access to the hosted model, ensuring secure usage and preventing unauthorized access.

In summary, Hugging Face serves as the backbone for the generation component of this RAG system. It enables efficient, cloud-hosted inference with a powerful LLM, simplifying the process of integrating cutting-edge NLP capabilities into the app. This allows the system to focus on retrieval (via FAISS) and context construction locally while offloading the computationally intensive generation tasks to Hugging Face’s infrastructure.

Don't know how to get an API key for HuggingFace? Check this guide: https://www.geeksforgeeks.org/how-to-access-huggingface-api-key/

1. **Ingest & Chunk** PDFs, text, or Markdown into overlapping passages  
2. **Embed** each chunk with SentenceTransformer  
3. **Index** embeddings in FAISS for sub-second top-k search  
4. **Retrieve** the most relevant chunks for a user query  
5. **Generate** an answer with Llama-2-13B (via Hugging Face Inference API) using retrieved context  
6. **Serve** an interactive Gradio UI for real-time querying  
7. **Deploy** on Google Colab or Hugging Face Spaces

---

## 🚀 Features

- **Universal document support**: PDF, TXT, MD  
- **Customizable chunk size & overlap**  
- **State-of-the-art embeddings** via [SentenceTransformers](https://github.com/UKPLab/sentence-transformers)  
- **High-speed retrieval** through FAISS  
- **Configurable generation**: temperature, top_k, max_new_tokens  
- **Interactive web UI** with Gradio  
- **Easy deployment** to Colab & Spaces  

---

## 🛠️ Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/rag-qa-system.git
   cd rag-qa-system
