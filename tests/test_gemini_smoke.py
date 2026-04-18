"""Integration test for Gemini API connectivity."""

import os

import pytest


@pytest.mark.integration
def test_gemini_api_connectivity() -> None:
    """Test that the Gemini API is accessible with the provided key.

    This test only runs if GEMINI_API_KEY is set in the environment.
    It makes a single minimal API call to verify the SDK is wired correctly.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    from google import genai
    from google.genai import types

    # Create client (auto-reads GEMINI_API_KEY from env)
    client = genai.Client()

    # Make a minimal API call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Say 'ok' and nothing else."],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=10,
        ),
    )

    # Verify we got a response
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0
    print(f"Gemini response: {response.text}")
