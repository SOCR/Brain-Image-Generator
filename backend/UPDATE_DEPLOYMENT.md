# How to Update Your Lightsail Deployment

## Quick Answer

**Q: Does uploading to ECR automatically update Lightsail?**  
**A: NO** - You need to create a new deployment after pushing to ECR.

**Q: Do I need to redeploy from scratch?**  
**A: NO** - Just run the deployment script and it creates a new deployment version.

---

## Update Process (Automated)

When you make changes to your backend code:

### Windows:
```bash
cd backend
deploy-to-lightsail.bat
```

### Linux/Mac:
```bash
cd backend
chmod +x deploy-to-lightsail.sh
./deploy-to-lightsail.sh
```

### What the Script Does:

1. ✅ Builds Docker image with CPU-only PyTorch
2. ✅ Pushes to ECR
3. ✅ Creates new Lightsail deployment
4. ✅ Waits for deployment to become active
5. ✅ Shows you the URL

**Time: ~5-10 minutes**

---

## Manual Process (If You Prefer)

### Step 1: Build and Push to ECR

```bash
cd backend

# Build with CPU-only PyTorch
docker build --platform linux/amd64 --provenance=false --sbom=false -t socr-image-gen-backend:latest .

# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 471112775480.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag socr-image-gen-backend:latest 471112775480.dkr.ecr.us-east-1.amazonaws.com/socr-image-gen-backend:latest
docker push 471112775480.dkr.ecr.us-east-1.amazonaws.com/socr-image-gen-backend:latest
```

### Step 2: Deploy to Lightsail

**Option A: Using AWS Console**
1. Go to Lightsail → Containers → socr-image-gen-backend
2. Click "Deployments" tab
3. Click "Create new deployment"
4. Click "Modify your deployment"
5. Everything should be the same, just click "Save and deploy"
6. Wait 3-5 minutes for deployment to complete

**Option B: Using AWS CLI**
```bash
# The deployment script does this for you, but if you want manual control:
aws lightsail create-container-service-deployment \
    --service-name socr-image-gen-backend \
    --region us-east-1 \
    --cli-input-json file://deployment.json
```

---

## How Lightsail Deployments Work

### Deployment Versions:
- Each time you deploy, Lightsail creates a **new deployment version**
- Old version keeps running until new version is healthy
- **Zero downtime** - automatic traffic switch
- Can rollback to previous versions if needed

### What Gets Updated:
- ✅ Docker image (pulls latest from ECR)
- ✅ Environment variables (if you change them)
- ✅ Container configuration (ports, health checks, etc.)

### What Stays the Same:
- ✅ Public URL (doesn't change)
- ✅ Service name
- ✅ Capacity/power settings

---

## Common Scenarios

### Scenario 1: Code Changes Only
**What to do:**
```bash
# Just run the deployment script
cd backend
deploy-to-lightsail.bat  # or .sh on Linux/Mac
```

### Scenario 2: Environment Variable Changes
**What to do:**
1. Update `.env` file in backend/
2. Run deployment script (it reads from `.env`)

### Scenario 3: Rollback to Previous Version
**What to do:**
```bash
# List deployment versions
aws lightsail get-container-service-deployments --service-name socr-image-gen-backend --region us-east-1

# Find the version number you want, then:
aws lightsail create-container-service-deployment \
    --service-name socr-image-gen-backend \
    --region us-east-1\
    --deployment <version-number>
```

Or use the AWS Console:
1. Go to Lightsail → Your service → Deployments
2. Click on a previous deployment
3. Click "Redeploy"

---

## Checking Deployment Status

### Check if deployment is complete:
```bash
aws lightsail get-container-services \
    --service-name socr-image-gen-backend \
    --region us-east-1 \
    --query "containerServices[0].currentDeployment.state" \
    --output text
```

States:
- `ACTIVATING` - Deployment in progress
- `ACTIVE` - Deployment complete and healthy
- `FAILED` - Something went wrong

### View logs:
```bash
aws lightsail get-container-log \
    --service-name socr-image-gen-backend \
    --container-name app \
    --region us-east-1
```

### Test health endpoint:
```bash
curl https://socr-image-gen-backend.nvtcqyjt0g9ej.us-east-1.cs.amazonlightsail.com/health
```

---

## Cost Note

Each deployment creates a **new version**, but you're only charged for:
- ✅ Running container(s)
- ✅ Data transfer

You're **NOT** charged extra for multiple deployment versions stored.

---

## Tips

1. **Always test locally first** before deploying
2. **Check logs** if deployment fails
3. **Keep old deployments** for easy rollback (Lightsail keeps last 10)
4. **Update frontend after backend** if there are API changes

---

## Troubleshooting

### Deployment stuck in ACTIVATING
- Check container logs for errors
- Verify health check passes: `/health` returns 200
- Check environment variables are set correctly

### 502/503 errors after deployment
- Wait 1-2 minutes for health checks to pass
- Check container logs for startup errors
- Verify Docker image built correctly

### Environment variables not updating
- Make sure to update `.env` file
- Run deployment script (it reads from `.env`)
- Check deployed config: AWS Console → Lightsail → Service → Current deployment

---

**That's it!** Your deployment process is now automated and simple. 🚀

