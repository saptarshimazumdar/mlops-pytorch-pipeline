#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "==> Applying Kubernetes namespace"
kubectl apply -f k8s/namespace.yaml

echo "==> Applying persistent volume claims"
kubectl apply -f k8s/pvc.yaml

echo "==> Applying training ConfigMap"
kubectl apply -f k8s/configmap.yaml

echo "==> Applying training Job"
kubectl apply -f k8s/training-job.yaml

echo "==> Applying serving Deployment"
kubectl apply -f k8s/serving-deployment.yaml

echo "==> Applying serving Service"
kubectl apply -f k8s/serving-service.yaml

echo "==> Checking job status"
kubectl get jobs -n ml-training

echo "==> Checking pod status"
kubectl get pods -n ml-training

echo "==> Checking serving status"
kubectl get deploy -n mlops-pytorch
kubectl get svc -n mlops-pytorch

echo "==> To follow training logs: kubectl logs -f job/ml-training-job -n ml-training"
echo "==> To check serving health: kubectl port-forward -n mlops-pytorch svc/pytorch-serving 8080:80"
