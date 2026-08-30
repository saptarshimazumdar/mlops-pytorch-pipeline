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

1. Create a virtual environment and install dependencies.
2. Run training with:
   `python src/train.py`
3. Start the serving API with:
   `uvicorn src.serve:app --reload --port 8000`
4. Build and deploy Docker images for training and inference via the Docker/Kubernetes manifests.

## Typical workflow

- Update model architecture in `src/model.py`
- Tune hyperparameters in `configs/training_config.yaml`
- Run `pytest` locally or through CI
- Package, train, and serve the model in containers and Kubernetes
