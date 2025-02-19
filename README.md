# Personal LLM API

A fine-tuned language model that answers questions about Alex Hunt, integrated with portfolio website.

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn app:app --reload
```

### Docker Build and Run
```bash
# Build the Docker image
docker build -t alexeugenehunt/personal-llm .

# Run the container
docker run -p 8000:8000 alexeugenehunt/personal-llm
```

### Manual Docker Hub Push
```bash
# Login to Docker Hub
docker login

# Push the image
docker push alexeugenehunt/personal-llm
```

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /ask`: Ask a question about Alex
  ```json
  {
    "text": "What are Alex's skills?",
    "conversation_id": "optional-conversation-id"
  }
  ```

## Deployment

This project is configured for automatic deployment using GitHub Actions and Render.

### GitHub Actions
The workflow automatically builds and pushes the Docker image on:
- Push to main branch
- New version tags
- Pull requests (build only)

### Render
Configured to pull and run the latest Docker image during deployment.
