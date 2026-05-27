import os
import requests
import sys

from dotenv import load_dotenv
from pathlib import Path

def check_credits() -> None:
    """Retrieves and displays the credit limit and remaining balance for the current API key.

    This function queries the OpenRouter API key endpoint to check the specific
    financial limits associated with the provided credentials.

    Args:
        None. The function retrieves the API key from the environment variables.

    Returns:
        None. Prints the formatted credit balance (Remaining / Total) in USD to the console.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in the environment or .env.")

    url = "https://openrouter.ai/api/v1/key"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    data = response.json().get("data", {})
    limit = data.get("limit")
    remaining = data.get("limit_remaining")

    print(f"Remaining Balance: ${remaining} / ${limit}")


def list_openrouter_models(model_name: str = ""):
    """Fetches available models from OpenRouter and prints their pricing in a formatted table.

    This function queries the OpenRouter API for all available models and displays their
    IDs alongside their prompt and completion costs scaled to per-million tokens.

    Args:
        `model_name : str`: A string to filter model IDs. Only models containing
            this string (case-insensitive) will be displayed. Defaults to an empty string.

    Returns:
        None. Prints a formatted table directly to the console.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in the environment or .env.")

    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    data = response.json().get("data", {})

    print(f"{'Model ID':<45} | {'Prompt ($/1M)':<15} | {'Completion ($/1M)':<15}")
    print("-" * 80)

    for el in data:
        if model_name.lower() in el["id"].lower():
            model_id = el["id"]
            p_prompt = float(el.get("pricing", {}).get("prompt", 0)) * 1e6
            p_comp = float(el.get("pricing", {}).get("completion", 0)) * 1e6

            print(f"{model_id:<45} | ${p_prompt:<14.5f} | ${p_comp:<14.5f}")


if __name__ == "__main__":

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    check_credits()

    filter_query = sys.argv[1] if len(sys.argv) > 1 else ""

    list_openrouter_models(filter_query)