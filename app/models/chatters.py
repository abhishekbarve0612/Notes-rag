from app.config import Settings


class OpenAIChatter:
    def __init__(self, settings: Settings):

        from openai import OpenAI
        if not settings.openai_api_key:
            raise ValueError("LLM_PROViDER=openai but OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_chat_model

    def answer(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user", "content": prompt,
            }]
        )

        return response.choices[0].message.content

class AnthropicChatter:
    def __init__(self, settings: Settings):
        from anthropic import Anthropic
        if not settings.anthropic_api_key:
            raise ValueError("LLM=anthropic but ANTHROPIC_API_KEY is empty")
        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_chat_model

    def answer(self, prompt: str) -> str:
        r = self._client.messages.create(
            model=self._model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text

class GeminiChatter:
    def __init__(self, settings: Settings):
        import google.generativeai as genai
        if not settings.gemini_api_key:
            raise ValueError("LLM=gemini but GEMINI_API_KEY is empty")
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_chat_model)

    def answer(self, prompt: str) -> str:
        return self._model.generate_content(prompt).text


class FakeChatter:                 # echoes prompt evidence; for tests + key-free dev
    def answer(self, prompt: str) -> str:
        head = prompt.strip().splitlines()[0][:80] if prompt.strip() else ""
        return f"[FAKE ANSWER] prompt started: {head!r}"