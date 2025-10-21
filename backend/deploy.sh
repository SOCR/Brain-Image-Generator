#!/bin/bash

# SOCR Image Generation Backend - Lightsail Deployment Script
# This script deploys the FastAPI backend to AWS Lightsail Container Service

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="socr-image-gen-backend"
CONTAINER_NAME="app"
REGION="us-east-1"  # Change if needed
POWER="micro"  # Options: nano ($7/mo), micro ($10/mo), small ($20/mo), medium ($40/mo), large ($80/mo), xlarge ($160/mo)
SCALE=1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SOCR Image Generation - Lightsail Deploy${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    echo "Please install it from: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if logged in
echo -e "\n${BLUE}Checking AWS credentials...${NC}"
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: Not logged into AWS CLI${NC}"
    echo "Please run: aws configure"
    exit 1
fi
echo -e "${GREEN}✓ AWS credentials verified${NC}"

# Check for environment variables
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create a .env file with your configuration"
    echo "You can copy .env.example and fill in your values"
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

echo -e "\n${BLUE}Step 1: Building Docker image...${NC}"
docker build -t $SERVICE_NAME:latest .
echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Check if container service exists
echo -e "\n${BLUE}Step 2: Checking if Lightsail container service exists...${NC}"
if aws lightsail get-container-services --service-name $SERVICE_NAME --region $REGION &> /dev/null; then
    echo -e "${GREEN}✓ Service exists${NC}"
else
    echo -e "${BLUE}Creating new Lightsail container service...${NC}"
    aws lightsail create-container-service \
        --service-name $SERVICE_NAME \
        --power $POWER \
        --scale $SCALE \
        --region $REGION
    
    echo -e "${GREEN}✓ Container service created${NC}"
    echo -e "${BLUE}Waiting for service to be active (this may take 2-3 minutes)...${NC}"
    
    # Wait for service to be active
    while true; do
        STATE=$(aws lightsail get-container-services --service-name $SERVICE_NAME --region $REGION --query 'containerServices[0].state' --output text)
        if [ "$STATE" == "ACTIVE" ]; then
            echo -e "${GREEN}✓ Service is now active${NC}"
            break
        fi
        echo "Current state: $STATE - waiting..."
        sleep 10
    done
fi

echo -e "\n${BLUE}Step 3: Pushing Docker image to Lightsail...${NC}"
aws lightsail push-container-image \
    --service-name $SERVICE_NAME \
    --label latest \
    --image $SERVICE_NAME:latest \
    --region $REGION

# Get the image reference
IMAGE_REF=$(aws lightsail get-container-images --service-name $SERVICE_NAME --region $REGION --query 'containerImages[0].image' --output text)
echo -e "${GREEN}✓ Image pushed: $IMAGE_REF${NC}"

echo -e "\n${BLUE}Step 4: Creating deployment configuration...${NC}"
# Create deployment JSON
cat > deployment.json <<EOF
{
  "containers": {
    "$CONTAINER_NAME": {
      "image": "$IMAGE_REF",
      "ports": {
        "8000": "HTTP"
      },
      "environment": {
        "SUPABASE_URL": "$SUPABASE_URL",
        "SUPABASE_KEY": "$SUPABASE_KEY",
        "FRONTEND_URL": "${FRONTEND_URL:-*}"
      }
    }
  },
  "publicEndpoint": {
    "containerName": "$CONTAINER_NAME",
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

echo -e "\n${BLUE}Step 5: Deploying container...${NC}"
aws lightsail create-container-service-deployment \
    --service-name $SERVICE_NAME \
    --region $REGION \
    --cli-input-json file://deployment.json

echo -e "${GREEN}✓ Deployment initiated${NC}"

# Clean up
rm deployment.json

echo -e "\n${BLUE}Step 6: Waiting for deployment to complete...${NC}"
echo "This may take 3-5 minutes..."

while true; do
    STATE=$(aws lightsail get-container-services --service-name $SERVICE_NAME --region $REGION --query 'containerServices[0].currentDeployment.state' --output text)
    if [ "$STATE" == "ACTIVE" ]; then
        echo -e "${GREEN}✓ Deployment is now active!${NC}"
        break
    fi
    echo "Current deployment state: $STATE - waiting..."
    sleep 15
done

# Get the public URL
echo -e "\n${BLUE}Getting service URL...${NC}"
URL=$(aws lightsail get-container-services --service-name $SERVICE_NAME --region $REGION --query 'containerServices[0].url' --output text)

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nYour API is now available at:"
echo -e "${BLUE}https://$URL${NC}"
echo -e "\nHealth check: ${BLUE}https://$URL/health${NC}"
echo -e "API docs: ${BLUE}https://$URL/docs${NC}"
echo -e "\n${BLUE}Update your Vercel frontend environment variable:${NC}"
echo -e "NEXT_PUBLIC_BACKEND_URL=https://$URL"
echo -e "\n${BLUE}To view service status:${NC}"
echo -e "aws lightsail get-container-services --service-name $SERVICE_NAME --region $REGION"

