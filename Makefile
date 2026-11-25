.PHONY: install train test lint api app docker-build docker-up

install:  ## Create venv and install everything
	uv venv --python 3.12 && uv pip install -e ".[api,app,dev]"

train:  ## Train model, write models/ and reports/figures/
	python -m churn.train

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

api:  ## http://localhost:8000/docs
	uvicorn api.main:app --reload

app:  ## http://localhost:8501
	streamlit run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:  ## API on :8000, UI on :8501
	docker compose up --build
