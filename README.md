# mlops-pytorch-pipeline

This project is a production-style PyTorch image classification pipeline covering the full ML lifecycle: local training, configuration-driven experimentation, Dockerized execution, CI validation, and Kubernetes deployment.

## Project structure

- `src/`: model code and training/serving entry points
- `configs/`: training configuration
- `docker/`: container build definitions for training and serving
- `k8s/`: Kubernetes manifests for namespace, jobs, deployment, service, config, and auto-scaling
- `requirements/`: dependency snapshots for train and serve stages
- `tests/`: validation tests for model behavior
- `.github/workflows/ci.yml`: CI pipeline for automated test execution

## Getting started

This project uses Python 3.11.

1. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements/train.txt
   ```
3. Run training with:
   ```bash
   python src/train.py
   ```
4. Run the model test:
   ```bash
   python -m pytest tests/test_model.py -q
   ```
5. Start the serving API with:
   ```bash
   python -m uvicorn src.serve:app --reload --port 8000
   ```
6. Build and deploy Docker images for training and inference via the Docker/Kubernetes manifests.

## Typical workflow

- Update model architecture in `src/model.py`
- Tune hyperparameters in `configs/training_config.yaml`
- Run tests locally or through CI
- Package, train, and serve the model in containers and Kubernetes
