import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import get_retriever
from agent import Agent

# Global state
retriever = None
agent = None
_initialized = False

def _ensure_initialized():
    global retriever, agent, _initialized
    if not _initialized:
        retriever = get_retriever()
        agent = Agent(retriever)
        _initialized = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_initialized()
    yield

app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if v[-1].role != "user":
            raise ValueError("last message must be from user")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global agent
    _ensure_initialized()
    
    # Convert pydantic models to dicts for agent
    messages = [m.model_dump() for m in request.messages]
    
    # Enforce turn limit awareness (max 8 turns total in conversation)
    if len(messages) > 8:
        return ChatResponse(
            reply="We've reached the conversation limit. Please start a new conversation if you need further assistance.",
            recommendations=[],
            end_of_conversation=True,
        )
    
    try:
        result = await agent.chat(messages)
        # Ensure strict schema compliance
        reply = result.get("reply", "")
        recommendations = result.get("recommendations", [])
        end_of_conversation = result.get("end_of_conversation", False)
        
        # Validate recommendations count
        if len(recommendations) > 10:
            recommendations = recommendations[:10]
        
        # Validate every recommendation URL is from catalog
        valid_names = {item["name"] for item in retriever.catalog}
        valid_urls = {item["url"] for item in retriever.catalog}
        filtered_recs = []
        for rec in recommendations:
            if rec["name"] in valid_names and rec["url"] in valid_urls:
                filtered_recs.append({
                    "name": rec["name"],
                    "url": rec["url"],
                    "test_type": rec["test_type"],
                })
        
        # If we intended to recommend but all were invalid, switch to clarify
        if recommendations and not filtered_recs:
            return ChatResponse(
                reply="I couldn't find matching assessments in the catalog. Could you provide more details about the role?",
                recommendations=[],
                end_of_conversation=False,
            )
        
        return ChatResponse(
            reply=reply,
            recommendations=filtered_recs,
            end_of_conversation=end_of_conversation,
        )
    except Exception as e:
        # Never crash the endpoint; return a safe response
        return ChatResponse(
            reply="I'm sorry, I encountered an issue. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )
