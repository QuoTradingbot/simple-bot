# Azure CLI Quick Reference for QuoTrading

## Common Commands

### Login & Account Management
```bash
# Login to Azure
az login

# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "My Subscription"

# Show current account
az account show
```

### Resource Management
```bash
# List all resource groups
az group list --output table

# Show resource group details
az group show --name quotrading-rg

# List all resources in a resource group
az resource list --resource-group quotrading-rg --output table

# Delete resource group (and all resources)
az group delete --name quotrading-rg --yes
```

### Web App Management
```bash
# List web apps
az webapp list --output table

# Show web app details
az webapp show --resource-group quotrading-rg --name quotrading-api

# Start web app
az webapp start --resource-group quotrading-rg --name quotrading-api

# Stop web app
az webapp stop --resource-group quotrading-rg --name quotrading-api

# Restart web app
az webapp restart --resource-group quotrading-rg --name quotrading-api

# Get web app URL
az webapp show --resource-group quotrading-rg --name quotrading-api --query defaultHostName -o tsv
```

### Configuration & Settings
```bash
# List app settings
az webapp config appsettings list --resource-group quotrading-rg --name quotrading-api --output table

# Set app setting
az webapp config appsettings set --resource-group quotrading-rg --name quotrading-api \
  --settings KEY=VALUE

# Delete app setting
az webapp config appsettings delete --resource-group quotrading-rg --name quotrading-api \
  --setting-names KEY1 KEY2

# Show startup command
az webapp config show --resource-group quotrading-rg --name quotrading-api \
  --query appCommandLine

# Update startup command
az webapp config set --resource-group quotrading-rg --name quotrading-api \
  --startup-file "gunicorn main:app --bind 0.0.0.0:8000"
```

### Logging & Monitoring
```bash
# Enable logging
az webapp log config --resource-group quotrading-rg --name quotrading-api \
  --docker-container-logging filesystem --level information

# Stream logs in real-time
az webapp log tail --resource-group quotrading-rg --name quotrading-api

# Download logs
az webapp log download --resource-group quotrading-rg --name quotrading-api \
  --log-file logs.zip

# Show log configuration
az webapp log show --resource-group quotrading-rg --name quotrading-api
```

### Database Management
```bash
# List PostgreSQL servers
az postgres flexible-server list --output table

# Show database server details
az postgres flexible-server show --resource-group quotrading-rg --name quotrading-db

# List databases
az postgres flexible-server db list --resource-group quotrading-rg \
  --server-name quotrading-db --output table

# Connect to database
az postgres flexible-server connect --name quotrading-db --admin-user quotradingadmin \
  --database-name quotrading_db

# Show firewall rules
az postgres flexible-server firewall-rule list --resource-group quotrading-rg \
  --name quotrading-db --output table

# Add firewall rule (allow specific IP)
az postgres flexible-server firewall-rule create --resource-group quotrading-rg \
  --name quotrading-db --rule-name MyIP --start-ip-address 1.2.3.4 --end-ip-address 1.2.3.4

# Start database server
az postgres flexible-server start --resource-group quotrading-rg --name quotrading-db

# Stop database server (saves costs)
az postgres flexible-server stop --resource-group quotrading-rg --name quotrading-db
```

### Deployment
```bash
# Deploy from local directory
az webapp up --resource-group quotrading-rg --name quotrading-api \
  --runtime PYTHON:3.11 --location eastus

# Configure GitHub deployment
az webapp deployment source config --resource-group quotrading-rg --name quotrading-api \
  --repo-url https://github.com/Quotraders/simple-bot --branch main --manual-integration

# Sync deployment
az webapp deployment source sync --resource-group quotrading-rg --name quotrading-api

# List deployment history
az webapp deployment list-publishing-profiles --resource-group quotrading-rg --name quotrading-api
```

### Scaling
```bash
# Scale up (change to higher tier)
az appservice plan update --resource-group quotrading-rg --name quotrading-plan --sku S1

# Scale out (add more instances)
az appservice plan update --resource-group quotrading-rg --name quotrading-plan \
  --number-of-workers 3

# Enable autoscale
az monitor autoscale create --resource-group quotrading-rg --name quotrading-autoscale \
  --resource quotrading-plan --resource-type Microsoft.Web/serverfarms \
  --min-count 1 --max-count 5 --count 2
```

### Troubleshooting
```bash
# Check if app is running
curl https://$(az webapp show --resource-group quotrading-rg --name quotrading-api \
  --query defaultHostName -o tsv)/

# View recent errors
az webapp log tail --resource-group quotrading-rg --name quotrading-api | grep -i error

# Check deployment status
az webapp deployment list --resource-group quotrading-rg --name quotrading-api

# SSH into container (if enabled)
az webapp ssh --resource-group quotrading-rg --name quotrading-api

# Test connection to database
az postgres flexible-server connect --name quotrading-db --admin-user quotradingadmin
```

### Cost Management
```bash
# Show current costs for resource group
az consumption usage list --resource-group quotrading-rg --output table

# Stop all services to save costs
az webapp stop --resource-group quotrading-rg --name quotrading-api
az postgres flexible-server stop --resource-group quotrading-rg --name quotrading-db

# Start all services
az webapp start --resource-group quotrading-rg --name quotrading-api
az postgres flexible-server start --resource-group quotrading-rg --name quotrading-db
```

### Backup & Recovery
```bash
# Create backup
az webapp config backup create --resource-group quotrading-rg --name quotrading-api \
  --backup-name mybackup --container-url "<storage-url>"

# List backups
az webapp config backup list --resource-group quotrading-rg --name quotrading-api

# Restore from backup
az webapp config backup restore --resource-group quotrading-rg --name quotrading-api \
  --backup-name mybackup
```

### Security
```bash
# Enable HTTPS only
az webapp update --resource-group quotrading-rg --name quotrading-api --https-only true

# Assign managed identity
az webapp identity assign --resource-group quotrading-rg --name quotrading-api

# Show managed identity
az webapp identity show --resource-group quotrading-rg --name quotrading-api

# Configure custom domain
az webapp config hostname add --resource-group quotrading-rg --name quotrading-api \
  --hostname api.quotrading.com

# Bind SSL certificate
az webapp config ssl bind --resource-group quotrading-rg --name quotrading-api \
  --certificate-thumbprint <thumbprint> --ssl-type SNI
```

## Useful Aliases

Add these to your `.bashrc` or `.zshrc`:

```bash
# QuoTrading Azure aliases
alias quo-logs='az webapp log tail --resource-group quotrading-rg --name quotrading-api'
alias quo-restart='az webapp restart --resource-group quotrading-rg --name quotrading-api'
alias quo-status='az webapp show --resource-group quotrading-rg --name quotrading-api --query state -o tsv'
alias quo-url='az webapp show --resource-group quotrading-rg --name quotrading-api --query defaultHostName -o tsv'
alias quo-settings='az webapp config appsettings list --resource-group quotrading-rg --name quotrading-api --output table'
```

## Environment Variables for Scripts

```bash
# Set these in your environment for easier scripting
export AZURE_RESOURCE_GROUP="quotrading-rg"
export AZURE_APP_NAME="quotrading-api"
export AZURE_DB_SERVER="quotrading-db"
export AZURE_LOCATION="eastus"

# Then use them
az webapp restart --resource-group $AZURE_RESOURCE_GROUP --name $AZURE_APP_NAME
```

## Tips

1. **Use `--output table` for readable output**
   ```bash
   az webapp list --output table
   ```

2. **Use `--query` to filter results**
   ```bash
   az webapp show --name quotrading-api --query defaultHostName -o tsv
   ```

3. **Use `--help` for command details**
   ```bash
   az webapp --help
   az webapp create --help
   ```

4. **Enable command completion**
   ```bash
   # Bash
   echo 'source /etc/bash_completion.d/azure-cli' >> ~/.bashrc
   
   # Zsh
   echo 'autoload -U +X bashcompinit && bashcompinit' >> ~/.zshrc
   echo 'source /etc/bash_completion.d/azure-cli' >> ~/.zshrc
   ```

5. **Use JMESPath for complex queries**
   ```bash
   az webapp list --query "[?state=='Running'].{Name:name, State:state}" --output table
   ```
