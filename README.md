# ChatbotQA

This is a chatbot application built with FastAPI. It uses a vector database for question answering and includes monitoring with Prometheus.

## Features

- FastAPI backend
- Question answering functionality
- Vector database integration (Qdrant)
- Prometheus for monitoring

## Getting Started

### Prerequisites

- Python 3.12
- Docker

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

Once the application is running, you can access the API at `http://localhost:8000`.

## API Endpoints

- `GET /`: Returns a "Hello World" message.
- `/api`: Main API routes.
- `/ingestion`: Data ingestion routes.

## Monitoring

The application is instrumented with Prometheus. You can access the metrics at the `/metrics` endpoint.

## Project Structure

```
.
├── app
│   ├── api
│   │   ├── ingestion.py
│   │   └── routes.py
├── docker-compose.yml
├── Dockerfile
├── main.py
├── prometheus.yml
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
    ├── core
    ├── services
    └── vector_db
```

