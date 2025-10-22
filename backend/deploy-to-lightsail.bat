@echo off
REM SOCR Image Generation - Complete Deployment to Lightsail via ECR
REM This script builds, pushes to ECR, and deploys to Lightsail

setlocal enabledelayedexpansion

REM Configuration
set SERVICE_NAME=socr-image-gen-backend
set CONTAINER_NAME=app
set REGION=us-east-1
set AWS_ACCOUNT_ID=471112775480
set ECR_REPO=%AWS_ACCOUNT_ID%.dkr.ecr.%REGION%.amazonaws.com/socr-image-gen-backend
set IMAGE_TAG=latest

echo ========================================
echo SOCR Image Generation - Deploy to Lightsail
echo ========================================

REM Check if .env file exists
if not exist ".env" (
    echo Error: .env file not found
    echo Please create a .env file with your configuration
    echo You can copy env.template and fill in your values
    exit /b 1
)

REM Load environment variables from .env
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
        set %%a=%%b
    )
)

REM Validate required environment variables
if "%SUPABASE_URL%"=="" (
    echo Error: SUPABASE_URL not set in .env
    exit /b 1
)
if "%SUPABASE_KEY%"=="" (
    echo Error: SUPABASE_KEY not set in .env
    exit /b 1
)

echo.
echo Step 1: Building Docker image with CPU-only PyTorch...
docker build --platform linux/amd64 --provenance=false --sbom=false -t %SERVICE_NAME%:%IMAGE_TAG% .
if %ERRORLEVEL% NEQ 0 (
    echo Error building Docker image
    exit /b 1
)
echo OK - Docker image built successfully

echo.
echo Step 2: Authenticating with ECR...
aws ecr get-login-password --region %REGION% | docker login --username AWS --password-stdin %ECR_REPO%
if %ERRORLEVEL% NEQ 0 (
    echo Error authenticating with ECR
    exit /b 1
)
echo OK - Authenticated with ECR

echo.
echo Step 3: Tagging image for ECR...
docker tag %SERVICE_NAME%:%IMAGE_TAG% %ECR_REPO%:%IMAGE_TAG%
echo OK - Image tagged

echo.
echo Step 4: Pushing image to ECR...
docker push %ECR_REPO%:%IMAGE_TAG%
if %ERRORLEVEL% NEQ 0 (
    echo Error pushing to ECR
    exit /b 1
)
echo OK - Image pushed to ECR

echo.
echo Step 5: Creating new Lightsail deployment...

REM Get the exact image digest from ECR
for /f "tokens=*" %%i in ('aws ecr describe-images --repository-name socr-image-gen-backend --region %REGION% --query "imageDetails[?imageTags[?@=='latest']].imageDigest" --output text') do set IMAGE_DIGEST=%%i

REM Use the tag-based reference (Lightsail prefers this)
set IMAGE_REF=%ECR_REPO%:%IMAGE_TAG%

REM Set default FRONTEND_URL if not set
if "%FRONTEND_URL%"=="" set FRONTEND_URL=*

REM Create deployment JSON
(
echo {
echo   "containers": {
echo     "%CONTAINER_NAME%": {
echo       "image": "%IMAGE_REF%",
echo       "ports": {
echo         "8000": "HTTP"
echo       },
echo       "environment": {
echo         "SUPABASE_URL": "%SUPABASE_URL%",
echo         "SUPABASE_KEY": "%SUPABASE_KEY%",
echo         "FRONTEND_URL": "%FRONTEND_URL%"
echo       }
echo     }
echo   },
echo   "publicEndpoint": {
echo     "containerName": "%CONTAINER_NAME%",
echo     "containerPort": 8000,
echo     "healthCheck": {
echo       "healthyThreshold": 2,
echo       "unhealthyThreshold": 2,
echo       "timeoutSeconds": 5,
echo       "intervalSeconds": 30,
echo       "path": "/health",
echo       "successCodes": "200-299"
echo     }
echo   }
echo }
) > deployment.json

aws lightsail create-container-service-deployment --service-name %SERVICE_NAME% --region %REGION% --cli-input-json file://deployment.json
if %ERRORLEVEL% NEQ 0 (
    echo Error creating deployment
    del deployment.json
    exit /b 1
)
echo OK - Deployment initiated

REM Clean up
del deployment.json

echo.
echo Step 6: Waiting for deployment to complete...
echo This may take 3-5 minutes...
timeout /t 10 /nobreak >nul

:wait_deployment
for /f "tokens=*" %%i in ('aws lightsail get-container-services --service-name %SERVICE_NAME% --region %REGION% --query "containerServices[0].currentDeployment.state" --output text') do set STATE=%%i
if not "%STATE%"=="ACTIVE" (
    echo Current deployment state: %STATE% - waiting...
    timeout /t 15 /nobreak >nul
    goto wait_deployment
)
echo OK - Deployment is now active!

REM Get the public URL
echo.
echo Getting service URL...
for /f "tokens=*" %%i in ('aws lightsail get-container-services --service-name %SERVICE_NAME% --region %REGION% --query "containerServices[0].url" --output text') do set URL=%%i

echo.
echo ========================================
echo DEPLOYMENT SUCCESSFUL!
echo ========================================
echo.
echo Your updated API is now available at:
echo !URL!
echo.
echo Health check: !URL!/health
echo API docs: !URL!/docs
echo.
echo Image deployed: %IMAGE_REF%
echo.

endlocal

