@echo off
REM SOCR Image Generation Backend - Lightsail Deployment Script (Windows)
REM This script deploys the FastAPI backend to AWS Lightsail Container Service

setlocal enabledelayedexpansion

REM Configuration
set SERVICE_NAME=socr-image-gen-backend
set CONTAINER_NAME=app
set REGION=us-east-1
set POWER=micro
set SCALE=1

echo ========================================
echo SOCR Image Generation - Lightsail Deploy
echo ========================================

REM Check if AWS CLI is installed
where aws >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: AWS CLI is not installed
    echo Please install it from: https://aws.amazon.com/cli/
    exit /b 1
)

REM Check if logged in
echo.
echo Checking AWS credentials...
aws sts get-caller-identity >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Not logged into AWS CLI
    echo Please run: aws configure
    exit /b 1
)
echo OK - AWS credentials verified

REM Check for .env file
if not exist ".env" (
    echo Error: .env file not found
    echo Please create a .env file with your configuration
    echo You can copy .env.example and fill in your values
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
echo Step 1: Building Docker image...
docker build -t %SERVICE_NAME%:latest .
if %ERRORLEVEL% NEQ 0 (
    echo Error building Docker image
    exit /b 1
)
echo OK - Docker image built successfully

REM Check if container service exists
echo.
echo Step 2: Checking if Lightsail container service exists...
aws lightsail get-container-services --service-name %SERVICE_NAME% --region %REGION% >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Creating new Lightsail container service...
    aws lightsail create-container-service --service-name %SERVICE_NAME% --power %POWER% --scale %SCALE% --region %REGION%
    echo OK - Container service created
    echo Waiting for service to be active (this may take 2-3 minutes)...
    
    :wait_active
    for /f "tokens=*" %%i in ('aws lightsail get-container-services --service-name %SERVICE_NAME% --region %REGION% --query "containerServices[0].state" --output text') do set STATE=%%i
    if not "%STATE%"=="ACTIVE" (
        echo Current state: %STATE% - waiting...
        timeout /t 10 /nobreak >nul
        goto wait_active
    )
    echo OK - Service is now active
) else (
    echo OK - Service exists
)

echo.
echo Step 3: Pushing Docker image to Lightsail...
aws lightsail push-container-image --service-name %SERVICE_NAME% --label latest --image %SERVICE_NAME%:latest --region %REGION%
if %ERRORLEVEL% NEQ 0 (
    echo Error pushing image to Lightsail
    exit /b 1
)

REM Get the image reference
for /f "tokens=*" %%i in ('aws lightsail get-container-images --service-name %SERVICE_NAME% --region %REGION% --query "containerImages[0].image" --output text') do set IMAGE_REF=%%i
echo OK - Image pushed: %IMAGE_REF%

echo.
echo Step 4: Creating deployment configuration...

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

echo.
echo Step 5: Deploying container...
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
echo Your API is now available at:
echo https://!URL!
echo.
echo Health check: https://!URL!/health
echo API docs: https://!URL!/docs
echo.
echo Update your Vercel frontend environment variable:
echo NEXT_PUBLIC_BACKEND_URL=https://!URL!
echo.
echo To view service status:
echo aws lightsail get-container-services --service-name %SERVICE_NAME% --region %REGION%
echo.

endlocal

