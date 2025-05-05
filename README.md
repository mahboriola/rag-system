# RAG System

A Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, OpenAI, and Qdrant vector database. This system allows you to ingest documents and ask questions about their content using advanced language models.

## Prerequisites

- Docker and Docker Compose
- OpenAI API key (Azure OpenAI Service supported)
- Azure Document Intelligence API key (for OCR capabilities)

## Setup Instructions

1. Clone the repository
2. Copy the `.env_sample` file to `.env`:
   ```bash
   cp .env_sample .env
   ```
3. Configure the environment variables in `.env` file (see [Environment Variables](#environment-variables) section below)
4. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

The services will be available at:
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8501
- Qdrant Dashboard: http://localhost:6333/dashboard

## Environment Variables

### OpenAI Configuration
- `OPENAI_TYPE`: The type of OpenAI service (azure/openai)
- `OPENAI_ENDPOINT`: Your OpenAI service endpoint URL
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_VERSION`: Azure OpenAI API version (e.g., "2024-10-21")
- `OPENAI_CHAT_MODEL`: The model to use for chat completion (e.g., "gpt-4")
- `OPENAI_EMBEDDING_MODEL`: The model to use for embeddings (e.g., "text-embedding-ada-002")

### Qdrant Configuration
- `QDRANT_ENDPOINT`: Qdrant server endpoint (default: "qdrant" when using Docker Compose)
- `QDRANT_PORT`: Qdrant server port (default: 6333)
- `QDRANT_COLLECTION_NAME`: Name of the collection to store document embeddings

### Azure Computer Vision (OCR)
- `AZURE_OCR_ENDPOINT`: Azure Document Intelligence API endpoint
- `AZURE_OCR_KEY`: Azure Document Intelligence API key

### Frontend Configuration
- `BACKEND_URL`: URL of the backend service (default: "http://localhost:8000" for local development)

## API Documentation
You can access the API documentation at:
- [openapi.json](openapi.json)
- [openapi.yaml](openapi.yaml)

## API Usage Examples

### 1. Document Ingestion

Upload a document to be processed and stored in the vector database:

```http
POST /ingest
Content-Type: multipart/form-data

files: [<document_file_1>, <document_file_2>]
```

Example Response:
```json
{
    "message": "Document processed successfully",
    "documents_indexed": 1,
    "total_chunks": 5
}
```

### 2. Question Answering

Ask questions about the ingested documents:

```http
POST /ask
Content-Type: application/json

{
    "question": "I want to know everything about the product XPTO"
}
```

Example Response:
```json
{
    "answer": "Based on the product documentation, the product specifications are...",
    "references": [
        "Dimensions...",
        "How to use the product...",
        "Safety instructions..."
    ]
}
```

## User Interface

The system includes a Streamlit-based frontend that provides a user-friendly interface for:

1. **Document Upload**: 
   - Easily upload PDF documents through a drag-and-drop interface
   - View processing status and confirmation of successful ingestion

2. **Interactive Chat**:
   - Chat-like interface for asking questions about your documents
   - Real-time responses from the RAG system
   - Display of relevant document references for each answer
   - Persistent chat history during the session

To access the frontend interface, simply navigate to http://localhost:8501 after starting the services.

## Architecture

The system consists of several components:

1. **Frontend Service**: Streamlit application providing the user interface
2. **Backend Service**: FastAPI application handling document ingestion and question answering
3. **Vector Database**: Qdrant for storing and retrieving document embeddings
4. **Document Processing Pipeline**: Handles document chunking, OCR, and embedding generation
5. **Question Answering Pipeline**: Processes questions using RAG techniques

## Data Flow

1. **Document Ingestion**:
   - Document upload → OCR (if needed) → Text chunking → Embedding generation → Vector storage

2. **Question Answering**:
   - Question → Embedding generation → Vector search → Context retrieval → LLM response generation


## Future Improvements
### Data Ingestion:
- Improve chunking strategies
- Register the Product Names in a database and retrive them to feed the LLM and improve the results when extracting the product name from the user query
- Extract metadata from each chunk to enhance search capabilities
- Improve document ingestion time
  
### Data Retrieval:
- Retrieve the product names from the database and pass them to the LLM to easily match the product name from the user query with the product name in the database
- Detect the intent of the user query and use it to filter the search in the vector database based on the keywords
  
### General:
- Add support for multiple LLM providers
- Add support for other OCR providers
- Add chat history as context to the LLM