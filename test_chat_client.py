import pytest
from chat_client import LocalChatClient

def test_chat_client_initialization():
    with pytest.raises(ValueError):
        LocalChatClient(api_key="")

def test_chat_response():
    client = LocalChatClient(api_key="test-token")
    response = client.chat("test message")
    assert isinstance(response, str)
    assert len(response) > 0 