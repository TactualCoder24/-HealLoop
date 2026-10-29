# Self-healing machines: one image runs the pipeline, the Streamlit dashboard and the MLflow UI.
#   docker compose up --build            -> pipeline (download -> train -> replay -> faults -> retrain -> report), then
#                                           dashboard on :8501 and MLflow UI on :5000
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONWARNINGS=ignore \
    MLFLOW_DISABLE_AGENT_HINT=1

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY selfheal selfheal
COPY configs configs
COPY app app
COPY tests tests
COPY pytest.ini .

EXPOSE 8501 5000
ENTRYPOINT ["python", "-m", "selfheal"]
CMD ["run-all"]
