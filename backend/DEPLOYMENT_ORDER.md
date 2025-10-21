# Deployment Order - Step by Step

This guide shows the **correct order** to deploy both backend and frontend to avoid the chicken-and-egg problem.

---

## Quick Summary

1. ✅ Deploy **Backend** first (with CORS = `*`)
2. ✅ Test backend locally
3. ✅ Deploy **Frontend** with backend URL
4. ✅ (Optional) Update backend with specific frontend URL

---

## Detailed Steps

### Step 1: Deploy Backend First (5-10 minutes)

**1.1 Create .env file**

```bash
cd backend
cp env.template .env
```

**1.2 Edit .env with your Supabase credentials**

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
FRONTEND_URL=*
```

> **Important**: Set `FRONTEND_URL=*` for now. This allows **all origins** temporarily.

**1.3 Deploy to Lightsail**

Windows:
```bash
deploy.bat
```

Linux/Mac:
```bash
chmod +x deploy.sh
./deploy.sh
```

**1.4 Save your backend URL**

At the end, you'll see:
```
Your API is now available at:
https://socr-image-gen-backend.xxxxx.us-east-1.cs.amazonlightsail.com
```

**Copy this URL!** You'll need it for the frontend.

---

### Step 2: Test Backend (2 minutes)

**2.1 Test health endpoint**

In your browser or using curl:
```bash
curl https://your-backend-url/health
```

Should return:
```json
{"status": "healthy", "service": "SOCR Image Generation API", "version": "1.0.0"}
```

**2.2 View API documentation**

Visit: `https://your-backend-url/docs`

You should see the interactive Swagger UI.

**2.3 Test locally with your frontend**

In your project root (not backend folder):

```bash
# Create/update .env.local
echo NEXT_PUBLIC_BACKEND_URL=https://your-backend-url > .env.local
```

Also add your Supabase vars:
```env
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url
```

Run frontend locally:
```bash
npm run dev
```

Test image generation - it should work! ✅

---

### Step 3: Deploy Frontend to Vercel (3-5 minutes)

**3.1 Push to GitHub (if not already done)**

```bash
git init
git add .
git commit -m "Ready for deployment"
git branch -M main
git remote add origin https://github.com/yourusername/socr-image-gen.git
git push -u origin main
```

**3.2 Import to Vercel**

1. Go to https://vercel.com
2. Click **"New Project"**
3. Import your GitHub repository
4. **Don't deploy yet!** First add environment variables...

**3.3 Add Environment Variables**

In Vercel project settings → Environment Variables, add:

```env
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_BACKEND_URL=https://your-lightsail-backend-url
```

> **Important**: Use the **anon key** for frontend, not the service_role key!

**3.4 Deploy**

Click **"Deploy"** and wait 2-3 minutes.

**3.5 Get your Vercel URL**

After deployment, you'll get:
```
https://your-app.vercel.app
```

Or whatever custom domain you configured.

---

### Step 4: (Optional) Lock Down CORS (2 minutes)

For **better security**, update backend to only allow your specific frontend:

**4.1 Update backend .env**

```bash
cd backend
```

Edit `.env`:
```env
FRONTEND_URL=https://your-app.vercel.app
```

**4.2 Redeploy backend**

Windows:
```bash
deploy.bat
```

Linux/Mac:
```bash
./deploy.sh
```

This restricts the backend to only accept requests from your Vercel domain.

---

## Summary

| Step | Action | Time | Result |
|------|--------|------|--------|
| 1 | Deploy backend with `FRONTEND_URL=*` | 5-10 min | Backend URL |
| 2 | Test backend + local frontend | 2 min | Confirm it works |
| 3 | Deploy frontend with backend URL | 3-5 min | Frontend URL |
| 4 | (Optional) Update backend CORS | 2 min | Tighter security |

**Total time: ~15-20 minutes** ⚡

---

## Why This Order?

### ❌ Wrong Order (Frontend First)
- Deploy frontend → Need backend URL (don't have it yet!) → ❌

### ✅ Correct Order (Backend First)
- Deploy backend with CORS=* → Get backend URL ✅
- Deploy frontend with backend URL → Get frontend URL ✅
- Update backend with frontend URL → Secure CORS ✅

---

## Testing Checklist

After deployment, test:

- [ ] Backend health check: `https://backend-url/health`
- [ ] Backend API docs: `https://backend-url/docs`
- [ ] Frontend loads: `https://frontend-url`
- [ ] User can sign up/login
- [ ] Can create a project
- [ ] Can generate images
- [ ] Images appear in library

---

## Troubleshooting

### "CORS error" in browser console

**If using CORS=\*:**
- Check `FRONTEND_URL=*` in backend .env
- Redeploy backend
- Clear browser cache

**If using specific URL:**
- Ensure exact match: `https://your-app.vercel.app` (no trailing slash!)
- Redeploy backend after changing .env
- Check browser console for actual vs allowed origin

### "Failed to fetch" errors

1. Check backend is accessible: `curl https://backend-url/health`
2. Check `NEXT_PUBLIC_BACKEND_URL` is set in Vercel
3. Verify URL has `https://` prefix
4. Check Lightsail service is running

### Backend won't deploy

1. Verify AWS credentials: `aws sts get-caller-identity`
2. Check Docker is running: `docker ps`
3. Verify .env file exists and has correct values
4. Check deployment script output for specific errors

---

## Need Help?

See [DEPLOYMENT.md](./DEPLOYMENT.md) for more details or [QUICKSTART.md](./QUICKSTART.md) for rapid deployment.

---

**Pro Tip**: Keep `FRONTEND_URL=*` if you're still testing. You can always lock it down later! 🚀


