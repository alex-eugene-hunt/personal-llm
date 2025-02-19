from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime
import redis
import json
from functools import lru_cache

app = FastAPI(title="AlexAI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_EXPIRATION = 3600  # 1 hour

# Models
class Question(BaseModel):
    text: str
    conversation_id: Optional[str] = None

class Response(BaseModel):
    answer: str
    confidence: float
    source: str
    timestamp: str
    conversation_id: str

# Load model and tokenizer (cached)
@lru_cache()
def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        "./fine_tuned_model",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model

@lru_cache()
def get_tokenizer():
    return AutoTokenizer.from_pretrained("./fine_tuned_model")

# Enhanced system prompt
SYSTEM_PROMPT = """You are Alex Hunt, a Software Engineer and Data Scientist based in San Francisco.
Answer the following question based on your personal data and experiences.
Provide accurate, concise responses while maintaining a natural conversational tone.

Context from previous messages: {context}

Q: {question}
A:"""

# Conversation history management
conversation_history = {}

def get_context(conversation_id: str, window_size: int = 3) -> str:
    if not conversation_id or conversation_id not in conversation_history:
        return ""
    recent_messages = conversation_history[conversation_id][-window_size:]
    return " ".join([f"Q: {m['question']} A: {m['answer']}" for m in recent_messages])

@app.post("/ask", response_model=Response)
async def ask_question(question: Question):
    try:
        # Check cache first
        cache_key = f"qa:{question.text}"
        cached_response = redis_client.get(cache_key)
        if cached_response:
            return Response(**json.loads(cached_response))

        # Get model and tokenizer
        model = get_model()
        tokenizer = get_tokenizer()

        # Get conversation context
        context = get_context(question.conversation_id) if question.conversation_id else ""
        
        # Prepare prompt
        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question.text
        )

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                do_sample=True,
                temperature=0.2,
                top_p=0.85,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        answer = response_text.split("A:")[-1].strip()
        
        # Calculate confidence score (simplified)
        confidence = min(0.95, 0.5 + len(answer.split()) / 50)  # Basic heuristic
        
        # Create response object
        response = Response(
            answer=answer,
            confidence=confidence,
            source="fine-tuned-model",
            timestamp=datetime.now().isoformat(),
            conversation_id=question.conversation_id or str(time.time())
        )

        # Update conversation history
        if response.conversation_id:
            if response.conversation_id not in conversation_history:
                conversation_history[response.conversation_id] = []
            conversation_history[response.conversation_id].append({
                "question": question.text,
                "answer": answer
            })

        # Cache the response
        redis_client.setex(
            cache_key,
            CACHE_EXPIRATION,
            json.dumps(response.dict())
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
