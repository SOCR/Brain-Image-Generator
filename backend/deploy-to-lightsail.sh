#!/bin/bash

# SOCR Image Generation - Complete Deployment to Lightsail via ECR
# This script builds, pushes to ECR, and deploys to Lightsail

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="socr-image-gen-backend"
CONTAINER_NAME="app"
REGION="us-east-1"
AWS_ACCOUNT_ID="471112775480"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/socr-image-gen-backend"
IMAGE_TAG="latest"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SOCR Image Generation - Deploy to Lightsail${NC}"
echo -e "${BLUE}========================================${NC}"

# Check for environment variables
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create a .env file with your configuration"
    echo "You can copy env.template and fill in your values"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Validate required environment variables
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo -e "${RED}Error: Missing required environment variables${NC}"
    echo "Please ensure SUPABASE_URL and SUPABASE_KEY are set in .env"
    exit 1
fi

echo -e "\n${BLUE}Step 1: Building Docker image with CPU-only PyTorch...${NC}"
docker build --platform linux/amd64 --provenance=false --sbom=false -t ${SERVICE_NAME}:${IMAGE_TAG} .
echo -e "${GREEN}✓ Docker image built successfully${NC}"

echo -e "\n${BLUE}Step 2: Authenticating with ECR...${NC}"
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}
echo -e "${GREEN}✓ Authenticated with ECR${NC}"

echo -e "\n${BLUE}Step 3: Tagging image for ECR...${NC}"
docker tag ${SERVICE_NAME}:${IMAGE_TAG} ${ECR_REPO}:${IMAGE_TAG}
echo -e "${GREEN}✓ Image tagged${NC}"

echo -e "\n${BLUE}Step 4: Pushing image to ECR...${NC}"
docker push ${ECR_REPO}:${IMAGE_TAG}
echo -e "${GREEN}✓ Image pushed to ECR${NC}"

echo -e "\n${BLUE}Step 5: Creating new Lightsail deployment...${NC}"

# Use the tag-based reference
IMAGE_REF="${ECR_REPO}:${IMAGE_TAG}"

# Set default FRONTEND_URL if not set
FRONTEND_URL=${FRONTEND_URL:-*}

# Create deployment JSON
cat > deployment.json <<EOF
{
  "containers": {
    "${CONTAINER_NAME}": {
      "image": "${IMAGE_REF}",
      "ports": {
        "8000": "HTTP"
      },
      "environment": {
        "SUPABASE_URL": "${SUPABASE_URL}",
        "SUPABASE_KEY": "${SUPABASE_KEY}",
        "FRONTEND_URL": "${FRONTEND_URL}"
      }
    }
  },
  "publicEndpoint": {
    "containerName": "${CONTAINER_NAME}",
    "containerPort": 8000,
    "healthCheck": {
      "healthyThreshold": 2,
      "unhealthyThreshold": 2,
      "timeoutSeconds": 5,
      "intervalSeconds": 30,
      "path": "/health",
      "successCodes": "200-299"
    }
  }
}
EOF

aws lightsail create-container-service-deployment \
    --service-name ${SERVICE_NAME} \
    --region ${REGION} \
    --cli-input-json file://deployment.json

echo -e "${GREEN}✓ Deployment initiated${NC}"

# Clean up
rm deployment.json

echo -e "\n${BLUE}Step 6: Waiting for deployment to complete...${NC}"
echo "This may take 3-5 minutes..."
sleep 10

while true; do
    STATE=$(aws lightsail get-container-services --service-name ${SERVICE_NAME} --region ${REGION} --query 'containerServices[0].currentDeployment.state' --output text)
    if [ "$STATE" == "ACTIVE" ]; then
        echo -e "${GREEN}✓ Deployment is now active!${NC}"
        break
    fi
    echo "Current deployment state: $STATE - waiting..."
    sleep 15
done

# Get the public URL
echo -e "\n${BLUE}Getting service URL...${NC}"
URL=$(aws lightsail get-container-services --service-name ${SERVICE_NAME} --region ${REGION} --query 'containerServices[0].url' --output text)

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nYour updated API is now available at:"
echo -e "${BLUE}$URL${NC}"
echo -e "\nHealth check: ${BLUE}$URL/health${NC}"
echo -e "API docs: ${BLUE}$URL/docs${NC}"
echo -e "\nImage deployed: ${IMAGE_REF}"
echo ""

