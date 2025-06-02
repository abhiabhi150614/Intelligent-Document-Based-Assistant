# Medical Chat bot or Medical Knowledge Retrieval System 

## Overview
This project is a **Medical Knowledge Retrieval System** designed to efficiently process and query a **1000-page medical encyclopedia**. It utilizes **natural language processing (NLP)** and **vector search techniques** to retrieve relevant information from large textual datasets.

The system loads medical documents, splits them into manageable chunks, generates embeddings using **sentence-transformers**, and stores them in a **FAISS** index for fast retrieval. Users can input medical queries, and the system will return the most relevant information along with a generated response using a **text generation model**.

## Features
- **Automated document processing**: Supports PDF, DOCX, and TXT formats.
- **Efficient vector search**: Uses **FAISS (Facebook AI Similarity Search)** for quick retrieval.
- **Advanced NLP models**:
  - Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Text generation: `google/flan-t5-base`
- **Conversational query system**: Provides detailed and structured answers based on retrieved medical knowledge.
- **Interactive CLI interface** for querying the knowledge base.

---

## Installation

To set up the environment, install the required dependencies using the following command:

```sh
pip install sentence-transformers==2.2.2 transformers==4.35.2 faiss-cpu==1.7.4 langchain==0.0.352 chromadb==0.4.17 langchain-community==0.0.14 pypdf==4.0.0 huggingface_hub==0.19.4
```

Ensure you have the required libraries installed before proceeding.

---

## Document Processing
### 1. Create Document Storage
Create a folder named `docs` where all medical documents will be stored.

```python
import os
os.makedirs('docs', exist_ok=True)
```

### 2. Load and Split Documents
This script loads documents from the `docs` folder, splits them into smaller chunks, and generates embeddings.

```python
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Load documents
documents = []
for file in os.listdir("docs"):
    file_path = os.path.join("docs", file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.endswith(".docx") or file.endswith(".doc"):
        loader = Docx2txtLoader(file_path)
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        continue
    documents.extend(loader.load())

# Split documents into smaller chunks
document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
document_chunks = document_splitter.split_documents(documents)
```

### 3. Generate and Store Embeddings
```python
import faiss

# Load sentence embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
document_texts = [doc.page_content for doc in document_chunks]
document_embeddings = np.array([embedding_model.encode(doc) for doc in document_texts])

# Store embeddings in FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)
faiss.write_index(index, 'faiss_index.index')

# Save document chunks
np.save('document_chunks.npy', document_texts)
print("Vector database created and stored successfully.")
```

---

## Querying the Knowledge Base

To interact with the medical knowledge base, we implement a **query function** that retrieves relevant document chunks and generates responses.

### 1. Load FAISS Index and Pretrained Model
```python
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load FAISS index
index = faiss.read_index('faiss_index.index')

# Load document chunks
with open('document_chunks.npy', 'rb') as f:
    document_texts = np.load(f, allow_pickle=True)

# Load sentence embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load text generation model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
```

### 2. Query Function
```python
def query_knowledge_base(query, top_n=3):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)
    
    results = [(document_texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    
    print("Top relevant document chunks:")
    for i, (doc, score) in enumerate(results):
        print(f"Rank {i+1}:\nDocument: {doc}\nScore: {score}\n")
    
    top_documents = " ".join([doc for doc, _ in results])
    prompt = f"Based on the following information:\n{top_documents}\n\nPlease provide a detailed medical answer to the question: {query}"
    
    result = pipe(prompt, max_length=200)
    return result[0]['generated_text']
```

### 3. Interactive Query Loop
```python
if __name__ == "__main__":
    while True:
        query = input("Enter your medical query (or type '1' to exit): ")
        if query == "1":
            break
        answer = query_knowledge_base(query, top_n=3)
        print("Answer:", answer)
```

---

## Optional: Alternative Query Processing
This alternative method follows a similar approach but uses a different embedding retrieval process.
```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document_embeddings = embeddings.embed_documents(document_texts)
document_embeddings_array = np.array(document_embeddings)
```

---

## Hugging Face Login (Optional)
If required, log in to **Hugging Face** to access additional models:
```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## Conclusion
This **Medical Knowledge Retrieval System** is designed to provide fast, relevant, and accurate medical information retrieval using **AI-powered NLP techniques**. It processes a large corpus of medical documents, enabling efficient querying and response generation for **doctors, researchers, and healthcare professionals**.

---

## Future Enhancements
- **Improve response generation** using a more advanced medical-specific NLP model.
- **Integrate a web-based UI** for an intuitive user experience.
- **Expand data sources** to include research papers, clinical trial data, and medical guidelines.
- **Enhance indexing and ranking** for better relevance and accuracy.

For any improvements or contributions, feel free to update the code and extend its functionalities! 
