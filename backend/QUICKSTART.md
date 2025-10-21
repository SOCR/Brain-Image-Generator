# Quick Start - Deploy in 5 Minutes

## Prerequisites
- AWS CLI installed and configured (`aws configure`)
- Docker installed and running
- Supabase account with project created

## Steps

### 1. Set up environment variables

```bash
cd backend
cp env.template .env
```

Edit `.env` with your Supabase credentials:
```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_service_role_key
FRONTEND_URL=*
```

### 2. Deploy to AWS Lightsail

**Windows:**
```bash
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Get your backend URL

After deployment completes (5-10 min), you'll see:
```
Your API is now available at:
https://socr-image-gen-backend.xxxxx.us-east-1.cs.amazonlightsail.com
```

### 4. Test it

Visit:
- `https://your-url/health` - Should return `{"status": "healthy"}`
- `https://your-url/docs` - Interactive API documentation

### 5. Deploy frontend to Vercel

1. Push code to GitHub
2. Import to Vercel: https://vercel.com/new
3. Add environment variables:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
   NEXT_PUBLIC_BACKEND_URL=https://your-lightsail-url
   ```
4. Deploy!

### 6. Update backend CORS

After Vercel deployment, update backend `.env`:
```env
FRONTEND_URL=https://your-app.vercel.app
```

Redeploy backend:
```bash
./deploy.sh  # or deploy.bat
```

## Done! 🎉

Your app is now live:
- Frontend: `https://your-app.vercel.app`
- Backend API: `https://your-lightsail-url`

## Cost

- AWS Lightsail: $10/month
- Vercel: Free
- Supabase: Free (or $25/month for Pro)

**Total: $10/month minimum**

## Need Help?

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions.

