import openai

class OpenApiModel:
    """
    OpenApiModel wraps API calls to a language model (e.g., OpenAI).
    """

    def __init__(self, model_type: str, api_key: str):
        self.model_type = model_type
        self.api_key = api_key
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the API client based on the model type.
        """
        if self.model_type == "openai":
            openai.api_key = self.api_key
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generate a response from the language model.

        :param prompt: The input text prompt for the model.
        :param max_tokens: The maximum number of tokens for the response.
        :return: The generated response text.
        """
        if self.model_type == "openai":
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4"
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            return response.choices[0].text.strip()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
