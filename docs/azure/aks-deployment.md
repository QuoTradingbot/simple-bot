# Azure Kubernetes Service (AKS) Deployment Guide

This guide provides instructions for deploying the QuoTrading Bot to Azure Kubernetes Service for production-grade deployments with advanced orchestration capabilities.

## Why AKS?

Choose AKS when you need:
- Advanced orchestration and scheduling
- Multi-region deployments
- Complex networking requirements
- Advanced monitoring and logging
- Integration with existing Kubernetes infrastructure

## Prerequisites

1. Azure CLI installed and authenticated
2. kubectl installed
3. Helm 3 installed
4. Basic knowledge of Kubernetes

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Azure AKS Cluster                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Ingress Controller                  │   │
│  │         (nginx/Azure Application Gateway)        │   │
│  └────────────────┬───────────────┬─────────────────┘   │
│                   │               │                      │
│         ┌─────────▼────┐  ┌──────▼────────┐            │
│         │   Service     │  │   Service     │            │
│         │ Signal Engine │  │ Trading Bots  │            │
│         └─────────┬────┘  └──────┬────────┘            │
│                   │               │                      │
│         ┌─────────▼────┐  ┌──────▼────────┐            │
│         │  Deployment   │  │  StatefulSet  │            │
│         │ Signal Engine │  │ Trading Bots  │            │
│         │  (3 replicas) │  │  (N replicas) │            │
│         └───────────────┘  └───────────────┘            │
│                                                           │
└─────────────────────────────────────────────────────────┘
                   │               │
          ┌────────▼───────────────▼────────┐
          │  External Azure Resources       │
          ├─────────────────────────────────┤
          │  - PostgreSQL Flexible Server   │
          │  - Redis Cache                  │
          │  - Azure Monitor                │
          │  - Key Vault                    │
          └─────────────────────────────────┘
```

## Step 1: Create AKS Cluster

```bash
# Set variables
RESOURCE_GROUP="quotrading-rg"
LOCATION="eastus"
AKS_CLUSTER="quotrading-aks"
ACR_NAME="quotradingacr"

# Create AKS cluster with managed identity
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER \
  --location $LOCATION \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --network-plugin azure \
  --network-policy azure \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 5

# Get AKS credentials
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER

# Verify connection
kubectl get nodes
```

## Step 2: Attach ACR to AKS

```bash
# Attach ACR to AKS cluster
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER \
  --attach-acr $ACR_NAME

# Verify ACR integration
az aks check-acr \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER \
  --acr $ACR_NAME.azurecr.io
```

## Step 3: Create Kubernetes Manifests

### Namespace

Create `kubernetes/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quotrading
  labels:
    name: quotrading
```

### Secrets

Create `kubernetes/secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: quotrading-secrets
  namespace: quotrading
type: Opaque
stringData:
  db-password: "YourSecurePassword123!"
  stripe-api-key: "sk_live_xxxxxxxxxxxxx"
  stripe-webhook-secret: "whsec_xxxxxxxxxxxxx"
  topstep-api-token: "your-api-token"
  topstep-username: "your-email@example.com"
```

### ConfigMap

Create `kubernetes/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quotrading-config
  namespace: quotrading
data:
  DB_HOST: "quotrading-db.postgres.database.azure.com"
  DB_NAME: "quotrading"
  DB_USER: "quotradingadmin"
  DB_SSL_MODE: "require"
  ENVIRONMENT: "production"
  QUOTRADING_API_URL: "http://signal-engine-service:8000"
```

### Signal Engine Deployment

Create `kubernetes/signal-engine-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: signal-engine
  namespace: quotrading
  labels:
    app: signal-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: signal-engine
  template:
    metadata:
      labels:
        app: signal-engine
    spec:
      containers:
      - name: signal-engine
        image: quotradingacr.azurecr.io/signal-engine:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: quotrading-config
              key: DB_HOST
        - name: DB_NAME
          valueFrom:
            configMapKeyRef:
              name: quotrading-config
              key: DB_NAME
        - name: DB_USER
          valueFrom:
            configMapKeyRef:
              name: quotrading-config
              key: DB_USER
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: quotrading-secrets
              key: db-password
        - name: STRIPE_API_KEY
          valueFrom:
            secretKeyRef:
              name: quotrading-secrets
              key: stripe-api-key
        - name: STRIPE_WEBHOOK_SECRET
          valueFrom:
            secretKeyRef:
              name: quotrading-secrets
              key: stripe-webhook-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: signal-engine-service
  namespace: quotrading
spec:
  selector:
    app: signal-engine
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

### Trading Bot StatefulSet

Create `kubernetes/trading-bot-statefulset.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: trading-bot
  namespace: quotrading
spec:
  serviceName: trading-bot
  replicas: 2
  selector:
    matchLabels:
      app: trading-bot
  template:
    metadata:
      labels:
        app: trading-bot
    spec:
      containers:
      - name: trading-bot
        image: quotradingacr.azurecr.io/trading-bot:latest
        env:
        - name: QUOTRADING_API_URL
          valueFrom:
            configMapKeyRef:
              name: quotrading-config
              key: QUOTRADING_API_URL
        - name: TOPSTEP_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: quotrading-secrets
              key: topstep-api-token
        - name: TOPSTEP_USERNAME
          valueFrom:
            secretKeyRef:
              name: quotrading-secrets
              key: topstep-username
        - name: BOT_ENVIRONMENT
          value: "production"
        - name: BOT_DRY_RUN
          value: "false"
        - name: BOT_INSTRUMENT
          value: "ES"
        - name: BOT_MAX_CONTRACTS
          value: "3"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"
        volumeMounts:
        - name: bot-logs
          mountPath: /app/logs
  volumeClaimTemplates:
  - metadata:
      name: bot-logs
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi
```

### Ingress

Create `kubernetes/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quotrading-ingress
  namespace: quotrading
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.quotrading.com
    secretName: quotrading-tls
  rules:
  - host: api.quotrading.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: signal-engine-service
            port:
              number: 8000
```

## Step 4: Install NGINX Ingress Controller

```bash
# Add nginx ingress helm repo
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install nginx ingress
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-health-probe-request-path"=/healthz
```

## Step 5: Install Cert-Manager (for SSL)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@quotrading.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Step 6: Deploy Application

```bash
# Apply all manifests
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/signal-engine-deployment.yaml
kubectl apply -f kubernetes/trading-bot-statefulset.yaml
kubectl apply -f kubernetes/ingress.yaml

# Check deployment status
kubectl get pods -n quotrading
kubectl get services -n quotrading
kubectl get ingress -n quotrading
```

## Step 7: Configure Azure Monitor

```bash
# Enable Azure Monitor for containers
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER \
  --addons monitoring
```

## Step 8: Set Up Horizontal Pod Autoscaling

Create `kubernetes/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: signal-engine-hpa
  namespace: quotrading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: signal-engine
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Apply:
```bash
kubectl apply -f kubernetes/hpa.yaml
```

## Monitoring and Logging

### View Logs

```bash
# Signal engine logs
kubectl logs -n quotrading -l app=signal-engine --tail=100 -f

# Trading bot logs
kubectl logs -n quotrading -l app=trading-bot --tail=100 -f

# All logs
kubectl logs -n quotrading --all-containers=true --tail=100 -f
```

### Monitor Pods

```bash
# Watch pods
kubectl get pods -n quotrading -w

# Describe pod
kubectl describe pod -n quotrading <pod-name>

# Get pod metrics
kubectl top pods -n quotrading
```

## Updating Deployments

```bash
# Update signal engine image
kubectl set image deployment/signal-engine \
  signal-engine=quotradingacr.azurecr.io/signal-engine:v2.0 \
  -n quotrading

# Rollback if needed
kubectl rollout undo deployment/signal-engine -n quotrading

# Check rollout status
kubectl rollout status deployment/signal-engine -n quotrading
```

## Security Best Practices

1. **Use Azure Key Vault with CSI Driver**:
```bash
# Install secrets store CSI driver
helm repo add secrets-store-csi-driver https://kubernetes-sigs.github.io/secrets-store-csi-driver/charts
helm install csi-secrets-store secrets-store-csi-driver/secrets-store-csi-driver --namespace kube-system
```

2. **Enable Pod Security Policies**
3. **Use Network Policies**
4. **Enable Azure AD Integration**
5. **Implement RBAC**

## Cost Optimization

1. Use node pools with different VM sizes
2. Enable cluster autoscaler
3. Use spot instances for non-critical workloads
4. Set appropriate resource requests and limits

## Troubleshooting

### Pod not starting

```bash
kubectl describe pod -n quotrading <pod-name>
kubectl logs -n quotrading <pod-name> --previous
```

### Service not accessible

```bash
kubectl get endpoints -n quotrading
kubectl get ingress -n quotrading
```

### Database connection issues

```bash
# Test from within the cluster
kubectl run -it --rm debug \
  --image=postgres:15 \
  --restart=Never \
  -- psql -h quotrading-db.postgres.database.azure.com -U quotradingadmin -d quotrading
```

## Clean Up

```bash
# Delete all resources
kubectl delete namespace quotrading

# Delete AKS cluster
az aks delete \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER \
  --yes --no-wait
```

## Next Steps

1. Set up CI/CD with Azure DevOps or GitHub Actions
2. Configure backup and disaster recovery
3. Implement blue-green deployments
4. Set up multi-region deployment
5. Configure advanced monitoring and alerting
