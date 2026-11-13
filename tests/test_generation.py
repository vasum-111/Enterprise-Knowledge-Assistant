import httpx
import pytest
from src.generation import generate


@pytest.mark.parametrize("citation,valid", [("access:0", True), ("invented:99", False)])
def test_generation_rejects_unknown_citations(monkeypatch, citation, valid):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "fake-test-key")

    def respond(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"answer":"Use the access portal.","citations":["'
                            + citation
                            + '"]}'
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    sources = [{"id": "access:0", "text": "Use the access portal."}]
    if valid:
        result = generate("How?", sources, client)
        assert result["verified_citations"]
        assert not result["semantic_grounding_verified"]
    else:
        with pytest.raises(ValueError, match="citations"):
            generate("How?", sources, client)
