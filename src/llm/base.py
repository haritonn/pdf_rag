from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, context_docs: List[str]) -> str: ...
