# SOCR Image Generation - Deployment Guide

This guide will walk you through deploying the SOCR Image Generation application to production.

## Architecture Overview

- **Frontend**: Next.js deployed on Vercel
- **Backend**: FastAPI in Docker on AWS Lightsail Container Service
- **Database & Storage**: Supabase (managed service)

---

## Cost Breakdown

### AWS Lightsail Container Service
- **Micro (recommended)**: $10/month - 1GB RAM, 0.25 vCPU, 20 containers/month
- **Small**: $20/month - 2GB RAM, 0.5 vCPU, 40 containers/month
- **Medium**: $40/month - 4GB RAM, 1 vCPU, 80 containers/month

### Vercel
- **Hobby (Free)**: Perfect for this project
- **Pro ($20/month)**: Only if you need advanced features

### Supabase
- **Free tier**: 500MB database, 1GB storage (good to start)
- **Pro ($25/month)**: 8GB database, 100GB storage

**Total estimated cost: $10-35/month**

---

## Prerequisites

1. **AWS Account** with CLI configured
   ```bash
   aws configure
   ```

2. **Docker** installed and running
   ```bash
   docker --version
   ```

3. **Supabase Project** created at https://supabase.com
   - Note your `SUPABASE_URL` and `SUPABASE_KEY`

4. **GitHub Account** (for Vercel deployment)

---

## Step 1: Deploy Backend to AWS Lightsail

### 1.1 Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
cd backend
cp env.template .env
```

Edit `.env` with your actual values:

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
FRONTEND_URL=https://your-app.vercel.app
```

> **Important**: Use the **service_role** key from Supabase, not the anon key!

### 1.2 Run Deployment Script

**On Windows:**
```bash
cd backend
deploy.bat
```

**On Linux/Mac:**
```bash
cd backend
chmod +x deploy.sh
./deploy.sh
```

### 1.3 Wait for Deployment

The script will:
1. Build your Docker image
2. Create a Lightsail container service
3. Push the image to Lightsail
4. Deploy the container
5. Configure health checks

This takes about 5-10 minutes.

### 1.4 Get Your Backend URL

At the end of deployment, you'll see:

```
Your API is now available at:
https://socr-image-gen-backend.xxxxx.us-east-1.cs.amazonlightsail.com
```

**Save this URL!** You'll need it for the frontend.

### 1.5 Test the Backend

Visit in your browser:
- Health check: `https://your-backend-url/health`
- API docs: `https://your-backend-url/docs`

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Push to GitHub

If not already done:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/socr-image-gen.git
git push -u origin main
```

### 2.2 Import to Vercel

1. Go to https://vercel.com
2. Click **"Add New Project"**
3. Import your GitHub repository
4. Configure project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `./` (leave as default)
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

### 2.3 Set Environment Variables

In Vercel project settings, add these environment variables:

```env
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_BACKEND_URL=https://your-lightsail-backend-url
```

> **Important**: Use the **anon** key for the frontend!

### 2.4 Deploy

Click **"Deploy"** and wait 2-3 minutes.

### 2.5 Update Backend CORS

After deployment, update your backend's `.env` file:

```env
FRONTEND_URL=https://your-app.vercel.app
```

Redeploy the backend:

```bash
cd backend
./deploy.sh  # or deploy.bat on Windows
```

---

## Step 3: Configure Custom Domain (Optional)

### For Vercel (Frontend)

1. In Vercel project settings → Domains
2. Add your custom domain (e.g., `app.yourdomain.com`)
3. Follow DNS configuration instructions

### For Lightsail (Backend)

1. In AWS Console → Lightsail → Container Services
2. Click your service → Custom domains
3. Create a certificate and attach domain

---

## Updating Your Application

### Update Backend

1. Make your code changes
2. Run deployment script again:
   ```bash
   cd backend
   ./deploy.sh  # or deploy.bat
   ```

### Update Frontend

Just push to GitHub:
```bash
git add .
git commit -m "Your changes"
git push
```

Vercel auto-deploys on every push!

---

## Monitoring & Logs

### Backend Logs (Lightsail)

```bash
aws lightsail get-container-log \
  --service-name socr-image-gen-backend \
  --container-name app \
  --region us-east-1
```

Or visit: AWS Console → Lightsail → Container Services → Logs

### Frontend Logs (Vercel)

Visit: Vercel Dashboard → Your Project → Deployments → View Logs

### Supabase Logs

Visit: Supabase Dashboard → Your Project → Logs

---

## Cost Optimization Tips

1. **Start with Micro plan** ($10/month) - upgrade if needed
2. **Use Supabase free tier** initially
3. **Monitor usage** in AWS Lightsail dashboard
4. **Scale up only when necessary**

---

## Troubleshooting

### Backend not accessible

1. Check Lightsail service is running:
   ```bash
   aws lightsail get-container-services \
     --service-name socr-image-gen-backend \
     --region us-east-1
   ```

2. Check container logs for errors

3. Verify health check passes: `https://your-url/health`

### CORS errors

1. Ensure `FRONTEND_URL` is set correctly in backend `.env`
2. Redeploy backend after changing CORS settings
3. Clear browser cache

### Image generation fails

1. Check Supabase credentials are correct
2. Verify model files are included in Docker image
3. Check backend logs for Python errors

### Frontend can't connect to backend

1. Verify `NEXT_PUBLIC_BACKEND_URL` is set in Vercel
2. Check backend URL is accessible publicly
3. Ensure CORS is configured correctly

---

## Scaling Up

If traffic increases:

1. **Upgrade Lightsail plan**:
   ```bash
   aws lightsail update-container-service \
     --service-name socr-image-gen-backend \
     --power small \
     --scale 2
   ```

2. **Enable caching** (add Redis/CloudFront)

3. **Consider migration** to ECS Fargate for auto-scaling

---

## Deleting Resources

### Delete Backend

```bash
aws lightsail delete-container-service \
  --service-name socr-image-gen-backend \
  --region us-east-1
```

### Delete Frontend

In Vercel dashboard → Settings → Delete Project

---

## Support

- **AWS Lightsail**: https://lightsail.aws.amazon.com/
- **Vercel**: https://vercel.com/docs
- **Supabase**: https://supabase.com/docs

---

## Security Checklist

- [ ] Supabase RLS (Row Level Security) policies configured
- [ ] Backend uses service_role key (kept secret)
- [ ] Frontend uses anon key (safe to expose)
- [ ] CORS properly configured
- [ ] Environment variables not committed to Git
- [ ] HTTPS enabled (automatic with Lightsail & Vercel)

---

## Next Steps

1. Set up monitoring with AWS CloudWatch
2. Configure automated backups in Supabase
3. Set up CI/CD with GitHub Actions (optional)
4. Add custom domain names
5. Configure CDN for faster global access

Good luck with your deployment! 🚀


