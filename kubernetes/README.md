# QuoTrading Bot - Azure Container Deployment

This directory contains Kubernetes manifests for deploying the QuoTrading Bot to Azure Kubernetes Service (AKS).

## Quick Start

### Prerequisites

1. Azure CLI installed and authenticated
2. kubectl installed
3. An AKS cluster with ACR attached
4. PostgreSQL Flexible Server configured

### Deployment Steps

1. **Update secrets and configuration**:
   ```bash
   # Edit secrets.yaml with your actual credentials
   nano kubernetes/secrets.yaml
   
   # Edit configmap.yaml with your configuration
   nano kubernetes/configmap.yaml
   ```

2. **Apply manifests in order**:
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/secrets.yaml
   kubectl apply -f kubernetes/configmap.yaml
   kubectl apply -f kubernetes/signal-engine-deployment.yaml
   kubectl apply -f kubernetes/trading-bot-statefulset.yaml
   kubectl apply -f kubernetes/ingress.yaml
   kubectl apply -f kubernetes/hpa.yaml
   ```

3. **Verify deployment**:
   ```bash
   kubectl get pods -n quotrading
   kubectl get services -n quotrading
   kubectl get ingress -n quotrading
   ```

## Files Description

- **namespace.yaml**: Creates the quotrading namespace
- **secrets.yaml**: Stores sensitive information (passwords, API keys)
- **configmap.yaml**: Stores configuration values
- **signal-engine-deployment.yaml**: Deploys the signal engine API (3 replicas)
- **trading-bot-statefulset.yaml**: Deploys trading bots as StatefulSet
- **ingress.yaml**: Configures external access via NGINX ingress
- **hpa.yaml**: Horizontal Pod Autoscaler for signal engine

## Monitoring

```bash
# View logs
kubectl logs -n quotrading -l app=signal-engine --tail=100 -f
kubectl logs -n quotrading -l app=trading-bot --tail=100 -f

# Monitor pods
kubectl get pods -n quotrading -w

# Check metrics
kubectl top pods -n quotrading
```

## Updating

```bash
# Update signal engine
kubectl set image deployment/signal-engine \
  signal-engine=quotradingacr.azurecr.io/signal-engine:v2.0 \
  -n quotrading

# Update trading bot
kubectl set image statefulset/trading-bot \
  trading-bot=quotradingacr.azurecr.io/trading-bot:v2.0 \
  -n quotrading
```

## Scaling

```bash
# Scale signal engine manually
kubectl scale deployment/signal-engine --replicas=5 -n quotrading

# Scale trading bots
kubectl scale statefulset/trading-bot --replicas=3 -n quotrading
```

## Troubleshooting

```bash
# Describe pod
kubectl describe pod -n quotrading <pod-name>

# Get events
kubectl get events -n quotrading --sort-by='.lastTimestamp'

# Execute commands in pod
kubectl exec -it -n quotrading <pod-name> -- /bin/bash
```

## For More Information

See the full deployment guide at `docs/azure/aks-deployment.md`
