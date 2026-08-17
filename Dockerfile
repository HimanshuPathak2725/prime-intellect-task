# ──────────────────────────────────────────────────────────────────────────
# Stage 1: builder — installs the package in a slim Python image
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only the files needed for installation first (layer-cache friendly).
COPY pyproject.toml README.md ./
COPY mobile_ui_env/ ./mobile_ui_env/

# Install the package (no dev extras in production image).
RUN pip install --no-cache-dir -e "."

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — minimal image that can run eval and tests
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed package from builder.
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Copy remaining source files (tests, eval script).
COPY tests/ ./tests/
COPY run_eval.py ./

# Non-root user for security.
RUN useradd -m agentuser
USER agentuser

# Default: run heuristic eval on the eval split.
CMD ["python", "run_eval.py", "--agent", "heuristic", "--split", "eval", "--verbose"]

# ──────────────────────────────────────────────────────────────────────────
# Build instructions
# ──────────────────────────────────────────────────────────────────────────
# docker build -t mobile-ui-env .
# docker run --rm mobile-ui-env
#
# Run tests inside container:
# docker run --rm mobile-ui-env python -m pytest tests/ -v
#
# Run with LLM agent (pass API key at runtime, never bake it in):
# docker run --rm -e OPENAI_API_KEY=sk-... mobile-ui-env \
#   python run_eval.py --agent llm --model gpt-4o-mini
