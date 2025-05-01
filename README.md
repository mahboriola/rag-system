# RAG System

A Retrieval-Augmented Generation (RAG) system built with FastAPI, OpenAI, and Qdrant vector database. This system allows you to ingest documents and ask questions about their content using advanced language models.

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
- Qdrant Dashboard: http://localhost:6333

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

## Architecture

The system consists of several components:

1. **Backend Service**: FastAPI application handling document ingestion and question answering
2. **Vector Database**: Qdrant for storing and retrieving document embeddings
3. **Document Processing Pipeline**: Handles document chunking, OCR, and embedding generation
4. **Question Answering Pipeline**: Processes questions using RAG techniques

## Data Flow

1. **Document Ingestion**:
   - Document upload → OCR (if needed) → Text chunking → Embedding generation → Vector storage

2. **Question Answering**:
   - Question → Embedding generation → Vector search → Context retrieval → LLM response generation
