# 🔍 Generative AI RAG Pipeline (Haystack + Mistral + Pinecone)

An end-to-end Retrieval-Augmented Generation (RAG) system for intelligent document-based Q&A. Ingest documents from various formats, chunk and embed them semantically, index into Pinecone for fast retrieval, and generate grounded, context-aware answers using Mistral AI through a Haystack pipeline. Includes a minimal web app with server-rendered templates for easy interaction.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Haystack](https://img.shields.io/badge/Haystack-Pipeline_Framework-blue?style=for-the-badge&logo=haystack&logoColor=white)](https://haystack.deepset.ai/)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-Generative_Model-green?style=for-the-badge&logo=mistral&logoColor=white)](https://mistral.ai/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_Database-purple?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)

✨ **Features**

📄 **Document Ingestion**: Load PDFs, Markdown, and more from `data/`; configurable chunking, cleaning, and metadata extraction.

🔎 **Semantic Indexing**: Embed chunks with models like Sentence Transformers or Mistral Embed, and index into Pinecone (HNSW or Serverless) for efficient vector search.

🤖 **RAG Pipeline**: Retrieve top-k relevant passages via Haystack, compose context, and generate answers with Mistral—ensuring responses are grounded and cited.

🌐 **Simple Web UI**: Server-rendered HTML templates for querying the system and viewing answers with source citations.

📓 **Experimental Notebooks**: Jupyter notebooks in `notebook/` for data exploration, pipeline testing, and prototyping.

🧩 **Modular Architecture**: Core logic in `QASystem/` package for easy reuse, extension, or integration.

🗂️ **Project structure**
```
.
├─ QASystem/                  # Core pipeline code (ingest, retrieve, generate)
│  ├─ __init__.py
│  ├─ ingest.py               # Load → split → embed → upsert
│  ├─ retriever.py            # Pinecone client wrapper
│  ├─ pipeline.py             # Haystack Nodes / Pipeline wiring
│  ├─ generator.py            # Mistral client / prompt templates
│  └─ utils.py                # Text cleaning, chunking helpers
├─ data/                      # Source documents for indexing (PDF/Markdown/etc.)
├─ templates/                 # Jinja2/HTML templates for the web UI
├─ notebook/                  # Exploration and scratch work
├─ app.py                     # Demo server entrypoint
├─ requirementes.txt          # Python dependencies (note: file name has a typo)
├─ setup.py                   # Package metadata / editable install
└─ LICENSE                    # MIT License
```

🚀 **Quickstart**

**Prerequisites**
- Python 3.8 or higher
- API keys for Mistral AI and Pinecone
- Documents (PDFs, Markdown) to ingest

**1) Environment setup**
```bash
git clone https://github.com/AbdullahRasheed45/Generative_AI-RAG-With-Haystack-MistralAI-Pinecone.git
cd Generative_AI-RAG-With-Haystack-MistralAI-Pinecone

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies (note: file is named requirementes.txt)
pip install -r requirementes.txt
pip install -e .            # Optional: Install as editable package for development
```

**2) Configuration**
```bash
# Create environment file
cat > .env << EOF
# Mistral AI Configuration
MISTRAL_API_KEY=your_mistral_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=ragsample
PINECONE_CLOUD=aws          # Optional for serverless
PINECONE_REGION=us-east-1   # Optional for serverless

# Pipeline Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
CHUNK_SIZE=512
EOF
```

**3) Add documents and build index**
```bash
# Place your documents in data/ (e.g., PDFs or Markdown files)

# Ingest and index (adjust command based on your setup)
python -m QASystem.ingest --data ./data --index $PINECONE_INDEX
```

**4) Run the application**
```bash
# For FastAPI (using uvicorn)
uvicorn app:app --reload

# Or for Flask/simple script
python app.py

# Access the web UI at http://127.0.0.1:8000 (or the printed URL)
# Submit questions via the form and view grounded responses
```

🧠 **System architecture**

**1. Document Ingestion Layer**
```python
# QASystem/ingest.py - Flexible document loader and preprocessor
def ingest_documents(data_path: str, index_name: str):
    """Load, clean, chunk, embed, and upsert documents"""
    
    # Load documents with metadata
    documents = load_documents(data_path)  # Supports PDF, MD, etc.
    
    # Preprocess: Clean text, extract pages/sources
    cleaned_docs = [clean_text(doc) for doc in documents]
    
    # Chunking with overlap
    chunks = chunk_documents(cleaned_docs, size=CHUNK_SIZE, overlap=0.1)
    
    # Embed and upsert to Pinecone
    embeddings = embed_chunks(chunks, model=EMBEDDING_MODEL)
    upsert_to_pinecone(embeddings, index_name, metadata=chunks.metadata)
```

**2. Retrieval Layer**
```python
# QASystem/retriever.py - Pinecone integration
class PineconeRetriever:
    """Semantic search wrapper"""
    
    def __init__(self, index_name: str):
        self.index = pinecone.Index(index_name)
        
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Embed query and fetch top-k matches"""
        query_embedding = embed_text(query, model=EMBEDDING_MODEL)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']  # With scores, metadata (source, page)
```

**3. Generation Layer**
```python
# QASystem/generator.py - Mistral-powered answer synthesis
class MistralGenerator:
    """Generate grounded responses"""
    
    def __init__(self, api_key: str):
        self.client = MistralClient(api_key)
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Compose prompt and call Mistral"""
        prompt = f"""
        Question: {query}
        Context: {'\n\n'.join(contexts)}
        
        Answer the question using only the provided context. Cite sources by filename and page. If insufficient context, say "I don't have enough information."
        """
        response = self.client.chat.complete(prompt=prompt, model="mistral-large")
        return response.choices[0].message.content
```

**4. Pipeline Orchestration**
```python
# QASystem/pipeline.py - Haystack workflow
from haystack import Pipeline

def build_rag_pipeline(retriever, generator):
    """Wire retrieval and generation"""
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=generator, name="Generator", inputs=["Retriever"])
    return pipeline

# Usage in app.py
result = pipeline.run(query=user_query)
```

⚙️ **Configuration and customization**

**Pinecone Index Settings**:
```python
# Support for HNSW or Serverless
INDEX_CONFIG = {
    'metric': 'cosine',
    'dimension': 384,  # Match your embedding model (e.g., all-MiniLM-L6-v2)
    'pods': 1,         # For serverless, omit pods
    'pod_type': 's1'   # Adjust for scale
}
```

**Prompt and Response Customization**:
```python
# Configurable prompt templates
PROMPT_TEMPLATES = {
    'concise': "Answer briefly using context: {context}\nQuestion: {query}",
    'detailed': "Provide a detailed explanation with citations: {context}\nQuestion: {query}",
    'safety_first': "If context is irrelevant, respond: 'Insufficient information.'"
}

# Generation params
GENERATION_PARAMS = {
    'max_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9
}
```

🗣️ **Natural language query examples**

**Summary Queries**:
- *"Summarize the eligibility requirements from the uploaded policy PDFs."*
- *"List all definitions for the term ‘availability’ with citations."*

**Comparative Queries**:
- *"What are the key differences between documents A and B?"*
- *"Compare deployment steps for AWS vs. Azure based on the docs."*

**Instructional Queries**:
- *"According to the docs, how do I deploy the service on AWS?"*
- *"Explain the troubleshooting steps for common errors."*

**Exploratory Queries**:
- *"What best practices are mentioned for scaling the system?"*
- *"Extract key metrics from the performance reports."*

🔧 **Advanced features and extensions**

**Hybrid Search Integration**:
```python
# QASystem/retriever.py - Combine dense and sparse
class HybridRetriever:
    """Blend vector and keyword search"""
    
    def hybrid_search(self, query: str, top_k: int):
        dense_results = self.vector_search(query)
        sparse_results = self.keyword_search(query)  # e.g., BM25
        combined = rerank_results(dense_results + sparse_results)
        return combined[:top_k]
```

**Reranking Module**:
```python
# QASystem/utils.py - Post-retrieval refinement
def rerank_results(results: List[Dict], query: str) -> List[Dict]:
    """Use cross-encoder for better relevance"""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, res['metadata']['text']) for res in results]
    scores = model.predict(pairs)
    sorted_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [res for res, _ in sorted_results]
```

**Multi-Document Support**:
```python
# QASystem/ingest.py - Handle diverse formats
def load_documents(path: str) -> List[Dict]:
    """Load PDFs, MD, TXT with metadata"""
    loaders = {
        '.pdf': PDFLoader(),
        '.md': MarkdownLoader(),
        '.txt': TextLoader()
    }
    docs = []
    for file in os.listdir(path):
        ext = os.path.splitext(file)[1]
        if ext in loaders:
            docs.extend(loaders[ext].load(os.path.join(path, file)))
    return docs
```

🧪 **Development and testing framework**

**Interactive Notebooks**:
```python
# notebook/exploration.ipynb - Pipeline prototyping
def test_retrieval():
    """Validate top-k recall"""
    test_queries = [
        "Eligibility requirements",
        "AWS deployment steps"
    ]
    for query in test_queries:
        results = retriever.retrieve(query)
        print(f"Query: {query}")
        print(f"Top Result: {results[0]['metadata']['source']}")
        print("---")

def evaluate_rag_quality():
    """Assess answer grounding"""
    query = "Summarize policy eligibility"
    contexts = retriever.retrieve(query)
    answer = generator.generate_answer(query, contexts)
    print(answer)
```

**Performance Monitoring**:
```python
# QASystem/monitor.py - Track pipeline metrics
class RAGMonitor:
    """Log retrieval and generation stats"""
    
    def log_query(self, query: str, results: List[Dict], answer: str, time: float):
        self.metrics.increment('queries_total')
        self.metrics.histogram('response_time', time)
        self.metrics.gauge('avg_relevance', sum(r['score'] for r in results) / len(results))
```

🔒 **Production considerations**

**Security and Access Control**:
```python
# app.py - API key validation
class SecurityMiddleware:
    """Protect endpoints"""
    
    def validate_request(self, request: Request):
        if 'Authorization' not in request.headers:
            raise AuthenticationError("Missing API key")
        # Validate against env var or secret manager
```

**Scalability and Caching**:
```python
# QASystem/cache.py - Optimize frequent queries
class QueryCache:
    """Redis-backed caching"""
    
    def __init__(self):
        self.redis = redis.Redis()
        
    def get_cached_answer(self, query_hash: str) -> Optional[str]:
        return self.redis.get(query_hash)
    
    def set_cache(self, query_hash: str, answer: str, ttl: int = 3600):
        self.redis.set(query_hash, answer, ex=ttl)
```

🐛 **Troubleshooting guide**

**Common Configuration Issues**:
- **Pinecone Errors** → Verify API key, index name, region/cloud; match embedding dimensions.
- **Ingestion Fails** → Check document formats in data/; ensure loaders handle your files.
- **Empty Responses** → Increase TOP_K or check if docs were upserted (query Pinecone dashboard).

**Performance and Response Issues**:
- **Slow Retrieval** → Use serverless Pinecone or optimize chunk sizes.
- **Hallucinations** → Strengthen prompt grounding; add similarity thresholds.
- **App Not Starting** → Confirm uvicorn/Flask setup; fix requirementes.txt typo if needed.

**Development and Debugging**:
- **Import Errors** → Reinstall dependencies in venv.
- **Notebook Issues** → Restart kernel; install ipywidgets.
- **Model Timeouts** → Reduce max_tokens or use smaller Mistral variant.

📚 **Learning resources and roadmap**

**Technical Concepts Covered**:
- **RAG Patterns**: Ingestion, embedding, retrieval, generation.
- **Vector Databases**: Pinecone indexing and querying.
- **Orchestration**: Haystack pipelines for modular AI workflows.
- **Web Integration**: Simple apps with templates.

**Future Enhancement Ideas**:
- **Reranking/Hybrid Search**: Boost recall with BM25 + vectors.
- **Multi-Modal Support**: Add image/text extraction from PDFs.
- **Background Indexing**: Async jobs for large datasets.
- **Eval Framework**: Metrics for retrieval precision and answer quality.

📜 **License**

MIT License - see [LICENSE](LICENSE) file for complete terms.

## 📞 Connect & Support

<div align="center">

### 🚀 Ready to Build Advanced RAG Systems?

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://techvibes360.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullahrasheed-/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:abdullahrasheed45@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AbdullahRasheed45)

**Let's make document intelligence more accessible and powerful!**

</div>

---

*Built with ❤️ for developers interested in RAG and AI pipelines. Perfect for learning vector search, Haystack orchestration, and generative Q&A.*
