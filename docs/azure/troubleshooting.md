# Azure Container Deployment - Troubleshooting Guide

This guide helps resolve common issues when deploying QuoTrading Bot to Azure containers.

## Table of Contents

1. [Container Start Issues](#container-start-issues)
2. [Database Connection Issues](#database-connection-issues)
3. [Image Pull Errors](#image-pull-errors)
4. [Performance Issues](#performance-issues)
5. [Network Issues](#network-issues)
6. [Kubernetes Specific Issues](#kubernetes-specific-issues)
7. [Monitoring and Debugging](#monitoring-and-debugging)

## Container Start Issues

### Problem: Container fails to start

**Symptoms:**
- Pod status shows `CrashLoopBackOff` or `Error`
- Container Apps shows `Failed` status

**Diagnosis:**

For Container Apps:
```bash
# Check container logs
az containerapp logs show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --tail 100

# Check replica status
az containerapp replica list \
  --name signal-engine \
  --resource-group quotrading-rg
```

For AKS:
```bash
# Check pod status
kubectl describe pod -n quotrading <pod-name>

# Check logs
kubectl logs -n quotrading <pod-name> --previous
```

**Common Causes & Solutions:**

1. **Missing environment variables**
   - Check all required env vars are set
   - Verify secrets are properly configured

2. **Invalid database credentials**
   - Test database connection manually
   - Check firewall rules

3. **Port conflicts**
   - Verify target port matches container port
   - Check for port binding errors in logs

### Problem: Container starts but crashes immediately

**Diagnosis:**
```bash
# Check container exit code
kubectl get pod -n quotrading <pod-name> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'

# View last 20 lines of logs
kubectl logs -n quotrading <pod-name> --tail=20
```

**Solutions:**
- Exit code 137: Out of memory - increase memory limits
- Exit code 1: Application error - check application logs
- Exit code 139: Segmentation fault - check dependencies

## Database Connection Issues

### Problem: Cannot connect to PostgreSQL

**Diagnosis:**
```bash
# Test connectivity from within cluster
kubectl run -it --rm debug \
  --image=postgres:15 \
  --restart=Never \
  -- psql -h quotrading-db.postgres.database.azure.com \
         -U quotradingadmin \
         -d quotrading

# Check firewall rules
az postgres flexible-server firewall-rule list \
  --resource-group quotrading-rg \
  --name quotrading-db
```

**Common Solutions:**

1. **Firewall blocking connection**
   ```bash
   # Allow Azure services
   az postgres flexible-server firewall-rule create \
     --resource-group quotrading-rg \
     --name quotrading-db \
     --rule-name AllowAzureServices \
     --start-ip-address 0.0.0.0 \
     --end-ip-address 0.0.0.0
   ```

2. **Wrong connection string**
   - Verify DB_HOST includes `.postgres.database.azure.com`
   - Check DB_USER format (may need `user@server`)
   - Ensure DB_SSL_MODE is set to `require`

3. **Database doesn't exist**
   ```bash
   # Create database
   az postgres flexible-server db create \
     --resource-group quotrading-rg \
     --server-name quotrading-db \
     --database-name quotrading
   ```

### Problem: Connection timeout

**Solutions:**
```bash
# Increase connection timeout in ConfigMap
kubectl edit configmap quotrading-config -n quotrading

# Add:
DB_CONNECT_TIMEOUT: "30"
DB_COMMAND_TIMEOUT: "60"
```

## Image Pull Errors

### Problem: Failed to pull image from ACR

**Symptoms:**
- `ImagePullBackOff` or `ErrImagePull`
- Authentication errors

**Diagnosis:**
```bash
# Check ACR credentials
az acr credential show --name quotradingacr

# Test image pull
docker pull quotradingacr.azurecr.io/signal-engine:latest
```

**Solutions:**

1. **ACR not attached to AKS**
   ```bash
   az aks update \
     --resource-group quotrading-rg \
     --name quotrading-aks \
     --attach-acr quotradingacr
   ```

2. **Invalid credentials for Container Apps**
   ```bash
   # Update container app with fresh credentials
   ACR_USERNAME=$(az acr credential show --name quotradingacr --query username -o tsv)
   ACR_PASSWORD=$(az acr credential show --name quotradingacr --query passwords[0].value -o tsv)
   
   az containerapp update \
     --name signal-engine \
     --resource-group quotrading-rg \
     --registry-username $ACR_USERNAME \
     --registry-password $ACR_PASSWORD
   ```

3. **Image doesn't exist**
   ```bash
   # List images in ACR
   az acr repository list --name quotradingacr
   
   # List tags for specific image
   az acr repository show-tags \
     --name quotradingacr \
     --repository signal-engine
   ```

## Performance Issues

### Problem: High CPU usage

**Diagnosis:**
```bash
# Check metrics (AKS)
kubectl top pods -n quotrading

# Check metrics (Container Apps)
az monitor metrics list \
  --resource /subscriptions/{subscription-id}/resourceGroups/quotrading-rg/providers/Microsoft.App/containerApps/signal-engine \
  --metric "CpuPercentage"
```

**Solutions:**

1. **Increase CPU limits**
   ```bash
   # For Container Apps
   az containerapp update \
     --name signal-engine \
     --resource-group quotrading-rg \
     --cpu 1.0
   
   # For AKS - edit deployment
   kubectl edit deployment signal-engine -n quotrading
   # Update resources.limits.cpu
   ```

2. **Enable HPA (AKS only)**
   ```bash
   kubectl apply -f kubernetes/hpa.yaml
   ```

### Problem: High memory usage or OOM kills

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods -n quotrading

# Check for OOM kills
kubectl get events -n quotrading | grep OOM
```

**Solutions:**
```bash
# Increase memory limits
az containerapp update \
  --name signal-engine \
  --resource-group quotrading-rg \
  --memory 2.0Gi

# For AKS
kubectl edit deployment signal-engine -n quotrading
# Update resources.limits.memory
```

## Network Issues

### Problem: Service not accessible

**Diagnosis:**
```bash
# Check service endpoints (AKS)
kubectl get endpoints -n quotrading

# Check ingress
kubectl get ingress -n quotrading

# Test service internally
kubectl run -it --rm debug \
  --image=curlimages/curl \
  --restart=Never \
  -- curl http://signal-engine-service:8000/health
```

**Solutions:**

1. **No endpoints for service**
   - Check pod selector matches deployment labels
   - Verify pods are running and ready

2. **Ingress not working**
   ```bash
   # Check ingress controller
   kubectl get pods -n ingress-nginx
   
   # Describe ingress
   kubectl describe ingress quotrading-ingress -n quotrading
   ```

3. **DNS issues**
   ```bash
   # Test DNS resolution
   kubectl run -it --rm debug \
     --image=busybox \
     --restart=Never \
     -- nslookup signal-engine-service.quotrading.svc.cluster.local
   ```

### Problem: External access fails

**For Container Apps:**
```bash
# Check ingress configuration
az containerapp show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --query properties.configuration.ingress

# Verify FQDN
az containerapp show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --query properties.configuration.ingress.fqdn -o tsv
```

**For AKS:**
```bash
# Check ingress external IP
kubectl get ingress -n quotrading

# Check load balancer
kubectl get svc -n ingress-nginx
```

## Kubernetes Specific Issues

### Problem: Persistent volume issues

**Diagnosis:**
```bash
# Check PVC status
kubectl get pvc -n quotrading

# Describe PVC
kubectl describe pvc -n quotrading <pvc-name>
```

**Solutions:**
```bash
# Check storage classes
kubectl get storageclass

# Create storage class if missing
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: managed-premium
provisioner: disk.csi.azure.com
parameters:
  skuName: Premium_LRS
EOF
```

### Problem: StatefulSet pod not starting

**Diagnosis:**
```bash
kubectl describe statefulset trading-bot -n quotrading
kubectl get pods -n quotrading -l app=trading-bot
```

**Common issues:**
- PVC not binding
- Previous pod not fully terminated
- Anti-affinity rules preventing scheduling

## Monitoring and Debugging

### Enable detailed logging

**Container Apps:**
```bash
# Enable console logs
az containerapp logs show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --follow
```

**AKS:**
```bash
# Follow all container logs
kubectl logs -n quotrading -l app=signal-engine --all-containers=true -f

# Stern (if installed) - better log aggregation
stern -n quotrading signal-engine
```

### Check resource quotas

```bash
# Check resource quota
kubectl get resourcequota -n quotrading

# Check limit ranges
kubectl get limitrange -n quotrading
```

### Debug with temporary pod

```bash
# Run debug container
kubectl run -it --rm debug \
  --image=ubuntu \
  --restart=Never \
  --namespace=quotrading \
  -- bash

# Inside the container:
apt-get update && apt-get install -y curl postgresql-client
curl http://signal-engine-service:8000/health
psql -h quotrading-db.postgres.database.azure.com -U quotradingadmin -d quotrading
```

### Check Azure Monitor

```bash
# Get logs from Application Insights
az monitor app-insights query \
  --app quotrading-insights \
  --resource-group quotrading-rg \
  --analytics-query "requests | where timestamp > ago(1h) | limit 100"
```

### Common Health Check Commands

```bash
# Check all resources
kubectl get all -n quotrading

# Get events
kubectl get events -n quotrading --sort-by='.lastTimestamp' | tail -20

# Check node status
kubectl get nodes
kubectl describe node <node-name>

# Check cluster info
kubectl cluster-info
kubectl get componentstatuses
```

## Getting Help

If issues persist:

1. **Collect diagnostic information:**
   ```bash
   # For AKS
   kubectl get all -n quotrading -o yaml > quotrading-resources.yaml
   kubectl logs -n quotrading <pod-name> > pod-logs.txt
   kubectl describe pod -n quotrading <pod-name> > pod-describe.txt
   
   # For Container Apps
   az containerapp show \
     --name signal-engine \
     --resource-group quotrading-rg > containerapp-config.json
   ```

2. **Check Azure Service Health:**
   ```bash
   az monitor activity-log list \
     --resource-group quotrading-rg \
     --start-time 2024-01-01T00:00:00Z
   ```

3. **Review Azure documentation:**
   - [Container Apps troubleshooting](https://learn.microsoft.com/en-us/azure/container-apps/troubleshooting)
   - [AKS troubleshooting](https://learn.microsoft.com/en-us/azure/aks/troubleshooting)

4. **Contact support** with collected diagnostic information
