from collections.abc import Iterator
from typing import List

from ollama import Client


class OllamaProvider:
    def __init__(self, model_name, host="http://localhost:11434"):
        self.client = Client(host=host)
        self.model = model_name

    def _build_messages(self, prompt, context_docs) -> List[dict]:
        system = (
            "Ты ассистент, задача которого - ответить на вопрос, опираясь на источники. Отвечай только на основе данного контекста. "
            "Ответь на том же языке, что и пользователь. Если не удалось найти ответ в контексте - так и скажи. /no_think"
        )
        context_str = "\n\n---\n\n".join(context_docs)
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {prompt}",
            },
        ]

    def generate(self, prompt, context_docs):
        response = self.client.chat(
            model=self.model,
            messages=self._build_messages(prompt, context_docs),
            stream=False,
        )
        return response.message.content

    def stream(self, prompt, context_docs) -> Iterator[str]:
        for chunk in self.client.chat(
            model=self.model,
            messages=self._build_messages(prompt, context_docs),
            stream=True,
        ):
            yield chunk.message.content
