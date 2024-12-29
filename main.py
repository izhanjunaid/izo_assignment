import os
import shutil
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
# from slowapi import Limiter
# from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
import logging
from sentence_transformers import SentenceTransformer
import uvicorn
# from llama_index.…
import shutil
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
# from slowapi import Limiter
# from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
import logging
from sentence_transformers import SentenceTransformer
import uvicorn
# from llama_index.llms.ollama import Ollama  # Import Ollama LLM
from llama_index.llms.cerebras import Cerebras

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str = "1.0.0"

class QueryResponse(BaseModel):
    response: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    source_documents: Optional[list] = None

class UploadResponse(BaseModel):
    message: str
    document_id: str
    metadata: Dict[str, Any]
    processing_time: float

# Initialize rate limiter
# limiter = Limiter(key_func=get_remote_address)

# FastAPI app setup
app = FastAPI(
    title="Document QA API",
    description="API for document indexing and querying using LLM",
    version="1.0.0",
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)
# app.state.limiter = limiter

# Simple model options for embeddings
AVAILABLE_MODELS = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SIMILARITY_TOP_K = 5

class Query(BaseModel):
    text: str

class EmbeddingModel(BaseModel):
    model_name: str

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
Settings.embed_model = embed_model

# Ollama model setup
# ollama_llm = Ollama(model="llama3", base_url='http://localhost:11434', request_timeout=360.0)  # Ensure Ollama is used
llm = Cerebras(model="llama-3.3-70b", api_key="csk-f6rjp8v2xh6t9mkxvwymfye8c5wp4xd4nchpvntvf52xx8rm")# Ensure Cerebras is used
Settings.llm = llm  # Set Ollama as the default LLM

# Chroma setup for vector storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

async def clear_vector_store():
    """Clear the vector store by recreating the collection"""
    global chroma_client, chroma_collection, vector_store, storage_context
    try:
        logger.info("Clearing vector store...")
        # Delete and recreate collection
        try:
            chroma_client.delete_collection("my_collection")
        except Exception as e:
            logger.warning(f"Collection deletion warning (can be ignored): {str(e)}")
            
        chroma_collection = chroma_client.create_collection("my_collection")
        
        # Reinitialize vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Vector store cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the FastAPI app and set up vector store"""
    try:
        logger.info("Server starting...")

        # Clear storage on startup using clear_vector_store function
        await clear_vector_store()

        # Ensure Ollama model is ready for use
        global embed_model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during server initialization")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )

@app.post("/upload", response_model=UploadResponse)
# @limiter.limit("10/minute")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and index a document
    
    Args:
        file: Document file to be uploaded and indexed
    
    Returns:
        UploadResponse: Upload status and document metadata
    """
    start_time = time.time()
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file.filename}")
        documents = SimpleDirectoryReader(input_dir=temp_dir, filename_as_id=True).load_data()
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content could be extracted from the file")

        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            embed_model=embed_model
        )
        
        processing_time = time.time() - start_time
        logger.info(f"File processed successfully in {processing_time:.2f} seconds")

        return UploadResponse(
            message="Document uploaded and indexed successfully",
            document_id=documents[0].doc_id,
            metadata=documents[0].metadata,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query", response_model=QueryResponse)
# @limiter.limit("30/minute")
async def query_documents(query: Query):
    """
    Query indexed documents
    
    Args:
        query: Query text and optional parameters
    
    Returns:
        QueryResponse: Query results with confidence score and processing time
    """
    start_time = time.time()
    try:
        logger.info(f"Processing query: {query.text}")
        query_engine = VectorStoreIndex.from_vector_store(
            vector_store, 
            embed_model=embed_model
        ).as_query_engine(
            similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
            llm=llm
        )
        
        response = query_engine.query(query.text)
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=str(response),
            confidence_score=getattr(response, 'confidence', 0.0),
            processing_time=processing_time,
            source_documents=[doc.metadata for doc in getattr(response, 'source_documents', [])]
        )

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": str(e)
            }
        )

@app.post("/change-model")
# @limiter.limit("5/minute")
async def change_embedding_model(model_request: EmbeddingModel):
    """Change the embedding model with rate limiting"""
    global embed_model
    if model_request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        logger.info(f"Changing embedding model to {model_request.model_name}")
        new_model_name = AVAILABLE_MODELS[model_request.model_name]
        embed_model = HuggingFaceEmbedding(model_name=new_model_name)
        Settings.embed_model = embed_model

        logger.info(f"Model changed successfully to {model_request.model_name}")
        return {"message": f"Model changed to {model_request.model_name}"}
    except Exception as e:
        logger.error(f"Error changing model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error changing model")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__== "_main_":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )