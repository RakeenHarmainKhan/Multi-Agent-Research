"""Pydantic models for request/response validation."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Serializable message payload from the frontend."""
    role: Literal["user", "assistant", "system"]
    # Assistant reports routinely exceed 2k chars; prompts only use a truncated tail anyway.
    content: str = Field(..., min_length=1, max_length=20000)

class ResearchRequest(BaseModel):
    """Request model for initiating research."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1500, 
        description="Research question or topic"
    )
    thread_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    mode: Literal["auto", "research"] = Field(
        "auto", description="auto: classifier decides chat vs research; research: always run web research"
    )
    user_feedback: Optional[str] = Field(None, description="Feedback for report revision")
    messages: List[ConversationMessage] = Field(
        default_factory=list,
        max_length=50,
        description="Recent conversation history for stateless context",
    )


class FeedbackRequest(BaseModel):
    """Feedback model for human review flow."""
    feedback: str = Field(default="", max_length=2000)
    approved: bool = False


class ResearchResponse(BaseModel):
    """Response model for research results."""
    thread_id: str
    status: str
    report: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: Optional[float] = Field(
        None, description="USD; only set when LLM_*_COST_PER_1M_TOKENS are configured"
    )


class SystemStats(BaseModel):
    """System metrics for admin dashboard."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency: float
    total_tokens: int
    avg_tokens_per_query: float
    total_cost: Optional[float] = None
    queries_today: int

class HealthCheck(BaseModel):
    """API health status."""
    status: str
    version: str
    uptime: float = Field(description="Seconds since the backend process started")
    database: str
    llm_service: Literal["configured", "not_configured"]
    timestamp: datetime