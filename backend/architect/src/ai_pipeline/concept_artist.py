# backend/architect/src/ai_pipeline/concept_artist.py
"""
Stage A0: Concept Artist - 2D Concept Image Generation

Generates a concept image using Gemini Nano Banana Pro (gemini-3-pro-image-preview)
before 3D generation. The concept image serves as visual reference for all subsequent
stages (Blacksmith, Machinist, Artist).

This is the first step in the image-first generation pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types


# Lazy client initialization
_client = None


def _get_client():
    """Lazy initialization of Gemini client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    return _client


@dataclass
class ConceptResult:
    """Result from concept image generation."""
    
    image_bytes: bytes
    image_base64: str
    prompt_used: str
    aspect_ratio: str
    model: str = "gemini-3-pro-image-preview"


# Style presets for different asset types
STYLE_PRESETS: dict[str, str] = {
    "realistic": "photorealistic studio render with soft lighting, clean background, product photography style",
    "stylized": "stylized 3D render with cel-shading, vibrant colors, game asset style",
    "industrial": "industrial design render, technical drawing aesthetic, clean lines, neutral lighting",
    "fantasy": "fantasy concept art style, dramatic lighting, rich details, painterly quality",
    "scifi": "sci-fi concept art, futuristic design, metallic surfaces, neon accents",
    "organic": "organic design, natural materials, earthy tones, botanical influence",
    "military": "military equipment style, olive drab and tan colors, worn and functional aesthetic",
    "cyberpunk": "cyberpunk aesthetic, neon lights, dark atmosphere, high-tech low-life",
}


def _build_concept_prompt(
    user_prompt: str,
    style: str | None = None,
    category: str | None = None,
) -> str:
    """
    Build optimized prompt for concept image generation.
    
    Combines user prompt with style guidelines for best results.
    """
    # Get style description
    style_key = (style or "realistic").lower()
    style_desc = STYLE_PRESETS.get(style_key, STYLE_PRESETS["realistic"])
    
    # Category-specific angle guidance
    angle_guidance = "3/4 view angle showing form and depth"
    if category:
        cat_lower = category.lower()
        if cat_lower in ("weapon", "tool"):
            angle_guidance = "3/4 view showing full length, clear silhouette"
        elif cat_lower in ("vehicle", "machine"):
            angle_guidance = "3/4 front view showing main features"
        elif cat_lower in ("character", "creature"):
            angle_guidance = "front 3/4 view showing full figure"
        elif cat_lower in ("prop", "furniture"):
            angle_guidance = "eye-level 3/4 view"
    
    # Construct the full prompt
    concept_prompt = f"""Create a detailed concept art image of: {user_prompt}

Style: {style_desc}
Composition: {angle_guidance}
Background: Clean, neutral gradient background (do NOT include complex environments)
Lighting: Soft, even studio lighting to clearly show form and materials
Quality: High detail, sharp focus on the main subject

NEGATIVE CONSTRAINTS (CRITICAL):
- NO texture atlases, NO UV charts, NO Sprite sheets
- NO multiple angles or turnaround sheets (unless explicitly requested)
- NO blueprints, diagrams, or schematics
- NO text overlays or labels

Important: This is a single 3D object concept - show ONLY the described item in a single perspective view, centered in frame."""

    return concept_prompt


async def generate_concept_image(
    prompt: str,
    style: str | None = None,
    category: str | None = None,
    aspect_ratio: str = "1:1",
    timeout: float = 120.0,
) -> ConceptResult:
    """
    Generate a 2D concept image using Gemini Nano Banana Pro.
    
    Args:
        prompt: User's asset description
        style: Style preset key (realistic, stylized, industrial, etc.)
        category: Asset category for angle guidance (weapon, vehicle, etc.)
        aspect_ratio: Output aspect ratio (1:1, 16:9, 4:3, etc.)
        timeout: API timeout in seconds
        
    Returns:
        ConceptResult with image bytes and metadata
        
    Raises:
        RuntimeError: If generation fails after retries
    """
    client = _get_client()
    
    # Build optimized prompt
    concept_prompt = _build_concept_prompt(prompt, style, category)
    print(f"🎨 [ConceptArtist] Generating concept image...")
    print(f"   Style: {style or 'realistic'}")
    print(f"   Aspect: {aspect_ratio}")
    
    max_retries = 3
    last_error: Exception | None = None
    
    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}...")
            
            # Generate image using Gemini 3 Pro Image Preview
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=[concept_prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                        ),
                    ),
                ),
                timeout=timeout,
            )
            
            # Extract image from response
            image_bytes: bytes | None = None
            
            for part in response.parts:
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    break
            
            if image_bytes is None:
                raise RuntimeError("No image data in response")
            
            # Convert to base64 for storage/transmission
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            print(f"✅ [ConceptArtist] Generated concept image ({len(image_bytes)} bytes)")
            
            return ConceptResult(
                image_bytes=image_bytes,
                image_base64=image_base64,
                prompt_used=concept_prompt,
                aspect_ratio=aspect_ratio,
            )
            
        except asyncio.TimeoutError:
            last_error = RuntimeError(f"Image generation timed out after {timeout}s")
            print(f"⚠️ [ConceptArtist] Timeout on attempt {attempt + 1}")
            
        except Exception as e:
            last_error = e
            print(f"⚠️ [ConceptArtist] Error on attempt {attempt + 1}: {e}")
    
    raise RuntimeError(
        f"❌ [ConceptArtist] Failed after {max_retries} attempts: {last_error}"
    )


async def regenerate_concept_with_feedback(
    original_prompt: str,
    feedback: str,
    previous_image_base64: str | None = None,
    style: str | None = None,
    aspect_ratio: str = "1:1",
) -> ConceptResult:
    """
    Regenerate concept image with user feedback.
    
    Can optionally include the previous image as reference for iterative refinement.
    
    Args:
        original_prompt: Original user prompt
        feedback: User's feedback on what to change
        previous_image_base64: Previous concept image for reference (optional)
        style: Style preset
        aspect_ratio: Output aspect ratio
        
    Returns:
        New ConceptResult
    """
    client = _get_client()
    
    # Build refined prompt incorporating feedback
    refined_prompt = f"""Create a concept art image based on:

Original request: {original_prompt}

User feedback for this iteration: {feedback}

Apply the feedback to improve the concept. Maintain the same style and composition approach."""
    
    print(f"🎨 [ConceptArtist] Regenerating with feedback...")
    print(f"   Feedback: {feedback[:100]}...")
    
    # Build contents - optionally include previous image
    contents: list[Any] = [refined_prompt]
    
    if previous_image_base64:
        # Include previous image as reference
        previous_bytes = base64.b64decode(previous_image_base64)
        contents.append(
            types.Part.from_bytes(data=previous_bytes, mime_type="image/png")
        )
        print("   Including previous image as reference")
    
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            ),
            timeout=120.0,
        )
        
        # Extract image
        image_bytes: bytes | None = None
        for part in response.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                break
        
        if image_bytes is None:
            raise RuntimeError("No image data in response")
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        print(f"✅ [ConceptArtist] Regenerated concept image ({len(image_bytes)} bytes)")
        
        return ConceptResult(
            image_bytes=image_bytes,
            image_base64=image_base64,
            prompt_used=refined_prompt,
            aspect_ratio=aspect_ratio,
        )
        
    except Exception as e:
        raise RuntimeError(f"❌ [ConceptArtist] Regeneration failed: {e}")


def get_available_styles() -> dict[str, str]:
    """Return available style presets with descriptions."""
    return STYLE_PRESETS.copy()
