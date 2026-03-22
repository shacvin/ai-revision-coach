"""Unified LLM interface supporting Gemini (google.genai) and Azure OpenAI."""

import json

import config

_client = None


def _get_client():
    """Get or create Gemini client."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def generate(prompt: str, max_tokens: int = 2000, model: str = "generation") -> str:
    """
    Generate text using the configured LLM provider.

    Args:
        prompt: The prompt text
        max_tokens: Maximum tokens in response
        model: "generation" for main model, "evaluation" for lighter model

    Returns:
        Generated text string
    """
    if config.LLM_PROVIDER == "gemini":
        from google.genai import types

        model_name = (config.GEMINI_GENERATION_MODEL if model == "generation"
                      else config.GEMINI_EVALUATION_MODEL)
        client = _get_client()

        # Gemini 2.5 Flash uses thinking tokens that count against max_output_tokens.
        # Set output budget = requested + thinking headroom to avoid truncation.
        thinking_budget = 1024
        effective_max_tokens = max_tokens + thinking_budget + 512  # extra buffer

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=effective_max_tokens,
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        # Handle cases where response might be blocked
        if not response.candidates:
            raise ValueError("Gemini returned no candidates (possibly safety-filtered)")

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'unknown'
            raise ValueError(f"Gemini returned empty content (finish_reason={reason})")

        return candidate.content.parts[0].text

    elif config.LLM_PROVIDER == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_ENDPOINT,
        )
        response = client.chat.completions.create(
            model=config.AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")


def generate_json(prompt: str, max_tokens: int = 2000, model: str = "generation") -> dict | list:
    """Generate and parse JSON from LLM response."""
    text = generate(prompt, max_tokens, model)

    # Strip markdown code blocks if present
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            cleaned = part
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            if cleaned:
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Last resort: find JSON-like content
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}...")
