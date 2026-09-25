# =============================================================================================
# Dockerfile -- THE CPU IMAGE. This is the one that runs in production on AWS Lightsail.
# =============================================================================================
# There are now TWO Dockerfiles: this one and Dockerfile.gpu. Read Dockerfile.gpu's header for
# the full justification of why the split is two files rather than one file with a build ARG;
# the short version is that the GPU base image is a different operating system with no Python
# in it, so the two builds differ in their RUN steps and not merely in a value.
#
# WHAT THAT SPLIT COSTS THIS FILE: exactly one changed line, the COPY at the pip step below.
# Nothing else here moved. The four deploy scripts (deploy-to-lightsail.sh/.bat, deploy.sh/.bat)
# all invoke a bare `docker build ... .` with no `-f`, so `Dockerfile` is what they build by
# Docker's default-filename rule, and they were not touched. This image continues to serve the
# five production GAN models with the same pins, the same base and the same entrypoint.
#
# DO NOT "unify" this file with Dockerfile.gpu later without re-reading that header first --
# every edit to this file is an edit to the build that five working models depend on.
# ---------------------------------------------------------------------------------------------

# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
#
# BOTH FILES, AND THE SECOND ONE IS NOT OPTIONAL. requirements.txt's first directive is now
# `-r requirements-common.txt` (the shared pin list, factored out so the CPU and GPU targets
# cannot drift). pip resolves that include RELATIVE TO THE DIRECTORY OF THE FILE CONTAINING IT
# -- here /app -- and reads it off the filesystem at install time. If only requirements.txt
# were copied, the RUN below would fail with
#     ERROR: Could not open requirements file: [Errno 2] ... 'requirements-common.txt'
# and, because deploy-to-lightsail.sh runs under `set -e`, the deploy would abort at Step 1.
#
# requirements-gpu.txt is DELIBERATELY NOT copied here. It is unused by this image, and naming
# it would put it in this layer's cache key -- editing a GPU-only pin would then invalidate the
# CPU image's pip layer and force a full reinstall of torch on a deploy that changed nothing.
# It still reaches /app via the `COPY . .` below, which is fine: by then pip has already run.
COPY requirements.txt requirements-common.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY . .

# Create data directory for fallback storage
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

