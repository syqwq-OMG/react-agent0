"""A small offline model double. It is intentionally not a ReAct implementation."""
from copy import deepcopy

import pytest


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen_messages = []

    def generate(self, messages):
        self.seen_messages.append(deepcopy(messages))
        if len(self.seen_messages) > len(self.outputs):
            pytest.fail("Agent called the model more often than expected")
        output = self.outputs[len(self.seen_messages) - 1]
        if isinstance(output, Exception):
            raise output
        return output


@pytest.fixture
def scripted_model():
    return ScriptedModel


@pytest.fixture
def toy_context():
    # The public HotpotQA JSON represents a paragraph as [title, list-of-sentences].
    return [
        ["Ada Lovelace", ["Ada Lovelace worked on the Analytical Engine.", "Ada's father was Lord Byron."]],
        ["Lord Byron", ["Lord Byron was a British poet.", "He was born in London."]],
        ["Other Document", ["A completely unrelated paragraph about plants."]],
    ]


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Unit tests must never reach an API or silently incur cost."""
    import socket
    def blocked(*args, **kwargs):
        pytest.fail("Network disabled in unit tests; mock the transport instead")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
