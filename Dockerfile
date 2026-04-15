FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
CMD ["uv", "run", "gunicorn", "--workers", "1", "--threads", "8", "--timeout", "300", "--graceful-timeout", "30", "-b", "0.0.0.0:8080", "app:app"]
