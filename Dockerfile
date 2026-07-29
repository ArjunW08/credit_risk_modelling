# ============================================================
# Miniconda + uv — single-stage build
# ============================================================
FROM continuumio/miniconda3

WORKDIR /app

# Install curl for the health check
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create a conda environment named 'crm' with Python 3.11
RUN conda create -n crm python=3.11 -y && conda clean -afy

# Copy the uv binary from the official image for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project metadata and install dependencies into the conda env using uv
COPY pyproject.toml .
RUN uv pip install --python /opt/conda/envs/crm/bin/python --no-cache .

# Activate the conda environment by prepending it to PATH
ENV PATH="/opt/conda/envs/crm/bin:$PATH" \
    CONDA_DEFAULT_ENV="crm"

# Create a non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

# Copy application source code
# (data, models, logs, and notebooks are excluded by .dockerignore
#  and are bind-mounted at runtime instead)
COPY app.py .
COPY scripts/ scripts/

# Create mount-point directories so bind mounts work cleanly
RUN mkdir -p data models logs && \
    chown -R appuser:appuser /app

USER appuser

# Streamlit configuration: disable telemetry, set server defaults
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run the Streamlit inference app
CMD ["streamlit", "run", "app.py"]
