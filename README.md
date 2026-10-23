# ReAct From Scratch — minimal CS336-style starter

This is intentionally **almost empty**. It contains no Agent implementation, no prompt,
no retrieval, no controller, no model wrapper, and no scoring code.

## Included

- `adapters.py`: signatures only (`raise NotImplementedError`), as in CS336.
- `react_agent/__init__.py`: one empty package entrypoint. **Put all your own code here.**
- `tests/`: offline black-box interface tests with a tiny ScriptedModel test double.
- `pyproject.toml`, `.env.example`, `.gitignore`: environment setup only.
- `data/README.md`: points to the official dataset; does not ship answers.

## Environment

Use [uv](https://docs.astral.sh/uv/) to manage Python and project dependencies.
The project requires **Python 3.11 or newer**; the steps below use Python 3.11.

1. Install uv (macOS / Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation so `uv` is available on your `PATH`.
For Windows and other installation options, see the
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. From the project root, install Python and synchronize the environment:

```bash
uv python install 3.11
uv sync --python 3.11 --extra dev
```

`uv sync` creates `.venv`, installs the project in editable mode, and installs its
dependencies. `--extra dev` also installs the `dev` optional dependencies
(including pytest) defined in `pyproject.toml`. The first sync generates `uv.lock`;
subsequent syncs use it to keep dependency versions consistent.

3. Run the tests through uv:

```bash
uv run --extra dev python -m pytest -q
```

Use `uv run --extra dev python <script.py>` to run other Python scripts in the same
environment; manual activation is unnecessary. After changing dependencies in
`pyproject.toml`, run `uv sync --extra dev` again.

**Initial failing tests are EXPECTED**: that's your assignment. Implement in `react_agent/`
then connect your implementation through the thin functions in `adapters.py`.
Do not edit the tests. Do not implement the Agent in adapters.py.

## Provider examples (not implementations)

- Ollama: `ollama pull qwen2:7b`; base URL `http://localhost:11434/v1`, model `qwen2:7b`.
- DeepSeek: set `DEEPSEEK_API_KEY` in your local untracked `.env`; base URL
  `https://api.deepseek.com`; model `deepseek-flash` (verify current official docs).

The teaching protocol uses `tool_call`/`final` JSON as *plain generated text* so the
same controller can be evaluated on different providers without framework-specific tools.

## Honest benchmarking

Get HotpotQA from its official repository, write your own dataset split and evaluation
script, use the official scorer with a matching prediction+gold subset, and never put
gold answers/supporting facts in a prompt. Read the separate Assignment PDF/Markdown.
