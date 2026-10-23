"""A small offline model double. It is intentionally not a ReAct implementation."""
import pytest


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen_messages = []

    def generate(self, messages):
        self.seen_messages.append([dict(message) for message in messages])
        if len(self.seen_messages) > len(self.outputs):
            pytest.fail("Agent called the model more often than expected")
        return self.outputs[len(self.seen_messages) - 1]


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
