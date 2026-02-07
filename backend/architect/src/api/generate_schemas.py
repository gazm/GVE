# backend/architect/src/api/generate_schemas.py
"""
Pydantic request/response models for the AI Generation API.

Separated from generate.py to keep route handlers concise.
"""

from __future__ import annotations

from pydantic import BaseModel


class GenerateRequestAPI(BaseModel):
    """API request to generate an asset."""
    
    prompt: str
    category: str | None = None
    style_reference: str | None = None
    track_override: str | None = None  # "matter", "landscape", "audio"


class GenerateJobResponse(BaseModel):
    """Response with job ID for tracking."""
    
    job_id: str
    status: str


class GenerateStatusResponse(BaseModel):
    """Status of a generation job."""
    
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    asset_id: str | None = None
    result: dict | None = None
    error: str | None = None


class EstimateRequest(BaseModel):
    """Request for cost/time estimate."""
    
    category: str
    styles: list[str] = []
    prompt_length: int = 0


class EstimateResponse(BaseModel):
    """Cost and time estimate for generation."""
    
    cost_usd: float
    estimated_time_sec: int
    category: str


class MaterialSuggestion(BaseModel):
    """Material suggestion response."""
    
    materials: list[str]


# =============================================================================
# Concept Image Request/Response Models
# =============================================================================

class ConceptRequestAPI(BaseModel):
    """API request to generate a concept image."""
    
    prompt: str
    category: str | None = None
    style: str | None = None  # Style preset: realistic, stylized, industrial, etc.
    aspect_ratio: str = "1:1"


class ConceptJobResponse(BaseModel):
    """Response with concept job ID."""
    
    job_id: str
    status: str


class ConceptStatusResponse(BaseModel):
    """Status of a concept generation job."""
    
    job_id: str
    status: str  # "generating", "ready", "approved", "failed"
    concept_image: str | None = None  # Base64 image data
    prompt: str | None = None
    prompt_used: str | None = None  # Full prompt sent to model
    error: str | None = None


class ConceptApproveRequest(BaseModel):
    """Request to approve concept and proceed to 3D."""
    
    # Optional overrides for 3D generation
    track_override: str | None = None


class ConceptRegenerateRequest(BaseModel):
    """Request to regenerate concept with feedback."""
    
    feedback: str  # User's feedback on what to change
    use_previous_as_reference: bool = True  # Include previous image as reference


class ConceptApproveResponse(BaseModel):
    """Response after approving concept."""
    
    concept_job_id: str
    generation_job_id: str  # New job ID for 3D generation
    status: str
