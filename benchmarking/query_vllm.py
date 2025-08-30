import requests
import argparse
import sys

def query_model(prompt: str, model: str, max_tokens: int, url: str):
    """
    Sends a prompt to the vLLM server and prints the text response.
    """
    # Define the HTTP headers and JSON payload
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        
        # Extract and print the text from the first choice
        if data.get("choices") and len(data["choices"]) > 0:
            text_output = data["choices"][0].get("text", "Error: 'text' key not found in choice.").strip()
            print(text_output)
        else:
            print("Error: 'choices' field not found or is empty in the response.", file=sys.stderr)
            print("Full server response:", data, file=sys.stderr)

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the vLLM server: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple command-line client for a vLLM server."
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to send to the model."
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="The name of the model to query."
    )
    parser.add_argument(
        "-t", "--max-tokens",
        type=int,
        default=512,
        help="The maximum number of tokens to generate."
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="The URL of the vLLM completions endpoint."
    )

    args = parser.parse_args()

    prompt =  args.prompt  + 'Let\'s think step by step and output the final answer after \"####\".'

    
    query_model(
        prompt=prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        url=args.url
    )
