from types import SimpleNamespace

from src.evaluation import generation


def test_ollama_messages():
    provider = generation.OllamaProvider("test-model")

    messages = provider._build_messages(
        "What is this?",
        ["Document one", "Document two"],
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Document one" in messages[1]["content"]
    assert "What is this?" in messages[1]["content"]


class FakeClient:
    def __init__(self, host):
        self.host = host

    def chat(self, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(content="generated answer")
        )


def test_ollama_generation(monkeypatch):
    monkeypatch.setattr(generation, "Client", FakeClient)

    provider = generation.OllamaProvider("test-model")
    result = provider.generate("question", ["context"])

    assert result == "generated answer"
    assert provider.client.host == "http://localhost:11434"
