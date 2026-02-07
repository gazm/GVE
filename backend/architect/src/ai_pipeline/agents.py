# backend/architect/src/ai_pipeline/agents.py
"""
Base agent class and Gemini API integration.

Each agent has ONE focused task with constrained JSON output.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

# Initialize Gemini client on module load
_api_key = os.environ.get("GEMINI_API_KEY")
_client = None

def _get_client():
    """Lazy initialization of Gemini client."""
    global _client
    if _client is None:
        if not _api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=_api_key)
    return _client


@dataclass
class AgentContext:
    """Context passed to each agent in the pipeline."""
    
    user_prompt: str
    rag_context: dict[str, Any] = field(default_factory=dict)
    previous_outputs: dict[str, Any] = field(default_factory=dict)
    style_token: str | None = None
    constraints: dict[str, Any] | None = None


T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for AI agents.
    
    Each agent:
    - Has a single responsibility (geometry, materials, etc.)
    - Outputs structured JSON validated by Pydantic
    - Cannot modify outputs from previous stages
    """
    
    name: str = "BaseAgent"
    model_name: str = "gemini-3-pro-preview"  # Using latest Pro preview for better structured output
    temperature: float = 0.7
    max_retries: int = 3
    
    def __init__(self) -> None:
        self._client = _get_client()
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt defining agent role and constraints."""
        ...
    
    @abstractmethod
    def get_output_schema(self) -> type[T]:
        """Return the Pydantic model for output validation."""
        ...
    
    def _format_validation_error(self, error: Exception, original_prompt: str) -> str | None:
        """
        Format Pydantic ValidationError as clear instructions for LLM retry.
        
        Returns formatted prompt string, or None if error can't be formatted.
        """
        if not isinstance(error, ValidationError):
            return None
        
        error_list = error.errors()
        if not error_list:
            return None
        
        # Build clear error messages
        error_messages = []
        example_fix = None
        
        for err in error_list:
            loc = err.get("loc", ())
            msg = err.get("msg", "")
            input_value = err.get("input")
            
            # Format field path
            field_path = " → ".join(str(l) for l in loc)
            loc_str = str(loc).lower()
            
            # Special handling for common errors
            if "children" in loc_str and isinstance(input_value, str):
                error_messages.append(
                    f"❌ {field_path}: Got string '{input_value}' but must be an object.\n"
                    f"   Each child in 'children' array must be a complete object with: id, type, shape, params"
                )
                example_fix = {
                    "id": "example_id",
                    "type": "primitive",
                    "shape": "cylinder",
                    "params": {"radius": 0.5, "height": 1.0},
                    "lod_cutoff": 0
                }
            elif "subtract" in loc_str:
                # Machinist-specific: subtract field must be a dict, not a string
                error_messages.append(
                    f"❌ {field_path}: The 'subtract' field MUST be a dictionary/object.\n"
                    f"   Got: {type(input_value).__name__} = {repr(input_value)[:100]}\n"
                    f"   Expected: {{\"type\": \"primitive\", \"shape\": \"cylinder\", \"params\": {{...}}}}"
                )
                example_fix = {
                    "op": "subtract",
                    "target_node_id": "node_001",
                    "subtract": {
                        "type": "primitive",
                        "shape": "cylinder",
                        "params": {"radius": 0.1, "height": 0.3}
                    },
                    "lod_cutoff": 1
                }
            elif "add_operations" in loc_str and isinstance(input_value, str):
                # Machinist-specific: add_operations array items must be dicts
                error_messages.append(
                    f"❌ {field_path}: Each item in 'add_operations' must be an object.\n"
                    f"   Got string: '{input_value}'\n"
                    f"   Expected: Complete operation object with op, target_node_id, subtract, lod_cutoff"
                )
                example_fix = {
                    "op": "subtract",
                    "target_node_id": "node_001",
                    "subtract": {
                        "type": "primitive",
                        "shape": "cylinder",
                        "params": {"radius": 0.1, "height": 0.3}
                    },
                    "lod_cutoff": 1
                }
            elif "type" in msg.lower() and "dict" in msg.lower():
                error_messages.append(
                    f"❌ {field_path}: {msg}\n"
                    f"   Expected an object/dictionary, but got {type(input_value).__name__}: {repr(input_value)[:100]}"
                )
            elif "required" in msg.lower():
                error_messages.append(
                    f"❌ {field_path}: Missing required field\n"
                    f"   {msg}"
                )
            else:
                error_messages.append(f"❌ {field_path}: {msg}")
        
        # Build retry prompt
        retry_prompt = f"""{original_prompt}

[VALIDATION ERRORS - Please fix these issues:]

{chr(10).join(error_messages)}"""
        
        if example_fix:
            retry_prompt += f"""

[EXAMPLE OF CORRECT FORMAT:]
{json.dumps(example_fix, indent=2)}

[Remember: children arrays must contain complete objects, not strings or IDs.]"""
        
        return retry_prompt
    
    def build_prompt(self, context: AgentContext) -> tuple[str, str]:
        """
        Build system instruction and user prompt from context.
        
        Returns: (system_instruction, user_prompt)
        """
        system = self.get_system_prompt()
        
        # Inject RAG context placeholders
        if context.rag_context:
            if "{rag_context.api_spec}" in system:
                system = system.replace(
                    "{rag_context.api_spec}",
                    json.dumps(context.rag_context.get("api_spec", {}), indent=2)
                )
            if "{rag_context.examples}" in system:
                system = system.replace(
                    "{rag_context.examples}",
                    json.dumps(context.rag_context.get("examples", []), indent=2)
                )
            if "{rag_context.material_registry}" in system:
                system = system.replace(
                    "{rag_context.material_registry}",
                    json.dumps(context.rag_context.get("material_registry", {}), indent=2)
                )
        
        # Inject previous stage outputs
        for stage_name, output in context.previous_outputs.items():
            placeholder = f"{{stage_{stage_name}_json}}"
            if placeholder in system:
                system = system.replace(placeholder, json.dumps(output, indent=2))
        
        # Inject style token
        if context.style_token and "{user_style_token}" in system:
            system = system.replace("{user_style_token}", context.style_token)
        
        return system, context.user_prompt
    
    async def generate(self, context: AgentContext) -> T:
        """
        Generate output using Gemini API.
        
        Validates output against schema and retries on failure.
        """
        system_instruction, user_prompt = self.build_prompt(context)
        schema = self.get_output_schema()
        
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"  [{self.name}] Attempt {attempt + 1}/{self.max_retries} - Preparing schema...")
                
                # Convert Pydantic model to JSON schema dict
                # Gemini API requires dict format, not Pydantic model
                json_schema = schema.model_json_schema()
                
                # Clean schema for Gemini compatibility
                def clean_schema(obj):
                    if isinstance(obj, dict):
                        # Remove additionalProperties (Gemini doesn't support it)
                        obj = {k: v for k, v in obj.items() if k != "additionalProperties"}
                        
                        # Ensure array items have proper schema definition
                        if "items" in obj and isinstance(obj["items"], dict):
                            obj["items"] = clean_schema(obj["items"])
                        
                        # Ensure minItems is set for required arrays
                        if obj.get("type") == "array" and "minItems" not in obj:
                            # Check if this is a required field (has minLength in original)
                            if "minLength" in obj:
                                obj["minItems"] = obj.pop("minLength")
                        
                        # Recursively clean nested objects
                        return {k: clean_schema(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_schema(item) for item in obj]
                    return obj
                
                json_schema = clean_schema(json_schema)
                
                # Log cleaned schema for debugging (first attempt only)
                if attempt == 0:
                    import logging
                    logging.debug(f"[{self.name}] Cleaned JSON schema: {json.dumps(json_schema, indent=2)[:500]}...")
                
                print(f"  [{self.name}] Calling Gemini API (model: {self.model_name})...")
                
                # Use async client from google-genai with timeout
                try:
                    response = await asyncio.wait_for(
                        self._client.aio.models.generate_content(
                            model=self.model_name,
                            contents=user_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=self.temperature,
                                response_mime_type="application/json",
                                response_json_schema=json_schema,  # Use dict format
                            ),
                        ),
                        timeout=120.0  # 2 minute timeout
                    )
                    print(f"  [{self.name}] API call completed, parsing response...")
                except asyncio.TimeoutError:
                    raise RuntimeError(f"API call timed out after 120 seconds")
                except Exception as api_error:
                    print(f"  [{self.name}] API call error: {api_error}")
                    raise
                
                # Parse JSON response
                json_text = response.text
                data = json.loads(json_text)
                result = schema.model_validate(data)
                
                print(f"✅ {self.name} completed (attempt {attempt + 1})")
                return result
                
            except json.JSONDecodeError as e:
                last_error = e
                print(f"⚠️ {self.name} JSON parse error (attempt {attempt + 1}): {e}")
                # Add error context to prompt for next retry
                if attempt < self.max_retries - 1:
                    user_prompt = f"{context.user_prompt}\n\n[ERROR: Previous response was invalid JSON. Please ensure your response is valid JSON.]"
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                print(f"⚠️ {self.name} error (attempt {attempt + 1}): {error_msg}")
                
                # If it's a Pydantic validation error, include details in retry
                if attempt < self.max_retries - 1 and "validation error" in error_msg.lower():
                    # Extract validation error details and format clearly
                    error_instructions = self._format_validation_error(e, context.user_prompt)
                    if error_instructions:
                        user_prompt = error_instructions
                    else:
                        # Fallback to basic error message
                        error_details = error_msg
                        if isinstance(e, ValidationError):
                            error_list = e.errors()
                            error_details = json.dumps(error_list, indent=2)
                        
                        user_prompt = f"""{context.user_prompt}

[VALIDATION ERROR - Please fix the following issues:]
{error_details}

[Remember: Each child in children arrays must be a complete object with id, type, shape, params, etc. NOT a string.]"""
        
        raise RuntimeError(
            f"❌ {self.name} failed after {self.max_retries} attempts: {last_error}"
        )


class GeminiVisionAgent(BaseAgent[T]):
    """Agent that can process images using Gemini's vision capabilities."""
    
    model_name: str = "gemini-3-pro-preview"  # Supports vision and better structured output
    
    async def generate_with_image(
        self, 
        context: AgentContext, 
        image_path: Path
    ) -> T:
        """Generate output with image input."""
        system_instruction, user_prompt = self.build_prompt(context)
        schema = self.get_output_schema()
        
        # Load image
        image_data = image_path.read_bytes()
        
        last_error: Exception | None = None
        current_prompt = user_prompt  # Track prompt for retry modifications
        
        for attempt in range(self.max_retries):
            try:
                print(f"  [{self.name}] Vision attempt {attempt + 1}/{self.max_retries}...")
                
                # Convert Pydantic model to JSON schema dict
                json_schema = schema.model_json_schema()
                
                # Clean up schema for Gemini compatibility
                def clean_schema(obj):
                    if isinstance(obj, dict):
                        obj = {k: v for k, v in obj.items() if k != "additionalProperties"}
                        # Ensure array items have proper schema definition
                        if "items" in obj and isinstance(obj["items"], dict):
                            obj["items"] = clean_schema(obj["items"])
                        return {k: clean_schema(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_schema(item) for item in obj]
                    return obj
                json_schema = clean_schema(json_schema)
                
                # Use Part.from_bytes for image input
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self.model_name,
                        contents=[
                            current_prompt,
                            types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=self.temperature,
                            response_mime_type="application/json",
                            response_json_schema=json_schema,
                        ),
                    ),
                    timeout=120.0  # 2 minute timeout
                )
                
                # Parse response
                json_text = response.text
                data = json.loads(json_text)
                result = schema.model_validate(data)
                
                print(f"✅ {self.name} (vision) completed (attempt {attempt + 1})")
                return result
                
            except json.JSONDecodeError as e:
                last_error = e
                print(f"⚠️ {self.name} (vision) JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    current_prompt = f"{user_prompt}\n\n[ERROR: Previous response was invalid JSON. Please ensure your response is valid JSON.]"
                
            except asyncio.TimeoutError:
                last_error = RuntimeError("API call timed out after 120 seconds")
                print(f"⚠️ {self.name} (vision) timeout (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                print(f"⚠️ {self.name} (vision) error (attempt {attempt + 1}): {error_msg}")
                
                # If it's a Pydantic validation error, include details in retry
                if attempt < self.max_retries - 1 and "validation error" in error_msg.lower():
                    # Extract validation error details and format clearly
                    error_instructions = self._format_validation_error(e, user_prompt)
                    if error_instructions:
                        current_prompt = error_instructions
                    else:
                        # Fallback to basic error message
                        error_details = error_msg
                        if isinstance(e, ValidationError):
                            error_list = e.errors()
                            error_details = json.dumps(error_list, indent=2)
                        
                        current_prompt = f"""{user_prompt}

[VALIDATION ERROR - Please fix the following issues:]
{error_details}

[CRITICAL REMINDER for Machinist:]
- Each item in add_operations must be a complete object
- The "subtract" field must be a DICTIONARY like {{"type": "primitive", "shape": "cylinder", "params": {{...}}}}
- NOT a string like "subtract": "node_id" 
"""
        
        raise RuntimeError(
            f"❌ {self.name} (vision) failed after {self.max_retries} attempts: {last_error}"
        )
