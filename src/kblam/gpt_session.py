import argparse
import os
from openai import OpenAI

valid_models = ["gpt-4o", "text-embedding-3-large", "gpt-4", "gpt-3.5-turbo"]

class GPT:
    def __init__(
        self,
        model_name: str,
        endpoint_url: str, #This is api key, but called endpoint_url to have consistency with generic Azure class
        system_msg: str = "You are an AI assistant.",
        max_retries: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        seed: int = None,
    ):
        if model_name not in valid_models:
            raise ValueError(f"Invalid model: {model_name}. Valid models are: {valid_models}")

        self.client = OpenAI(api_key=endpoint_url)
        self.model_name = model_name
        self.system_msg = system_msg
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed

    def generate_response(self, prompt: str) -> str | None:
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": prompt},
        ]
        for _ in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    seed=self.seed,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Retrying due to error: {e}")
        return None

    def generate_embedding(self, text: str) -> list[float] | None:
        for _ in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Retrying due to error: {e}")
        return None

def parser_args():
    parser = argparse.ArgumentParser(description="GPT Session with OpenAI")
    parser.add_argument("--model_name", type=str, default="text-embedding-3-large", help="OpenAI model name")
    parser.add_argument("--prompt", type=str, default="Embedding text", help="Prompt or text to embed")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    gpt = GPT(model_name=args.model_name, endpoint_url=args.api_key)

    if "embedding" in args.model_name.lower():
        response = gpt.generate_embedding(args.prompt)
    else:
        response = gpt.generate_response(args.prompt)

    assert response is not None
    print(response)
