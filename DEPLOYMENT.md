# 🚀 Deployment Guide - NeuroScan AI

## Split Deployment: Vercel (Frontend) + Render (Backend)

---

## 📋 Prerequisites

- GitHub account with your project repository
- Vercel account (free) - [vercel.com](https://vercel.com)
- Render account (free) - [render.com](https://render.com)

---

## 🔧 Part 1: Deploy Backend to Render

### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### Step 2: Create New Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository: `brain-tumor-detection-basic`
3. Configure:
   - **Name**: `neuroscan-ai-backend`
   - **Region**: Oregon (US West)
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

### Step 3: Add Environment Variables (Optional)
In Render dashboard → Environment:
- `PYTHON_VERSION`: `3.11.0`
- `FRONTEND_URL`: (leave empty for now, we'll add after Vercel deployment)

### Step 4: Deploy
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for build
3. Copy your backend URL: `https://neuroscan-ai-backend.onrender.com`

---

## 🎨 Part 2: Deploy Frontend to Vercel

### Step 1: Create Vercel Account
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub

### Step 2: Import Project
1. Click **"Add New..."** → **"Project"**
2. Import `brain-tumor-detection-basic` repository
3. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `dist` (auto-detected)

### Step 3: Add Environment Variable
In **Environment Variables** section:
- **Key**: `VITE_API_URL`
- **Value**: `https://neuroscan-ai-backend.onrender.com` (your Render backend URL)

### Step 4: Deploy
1. Click **"Deploy"**
2. Wait 2-3 minutesz
3. Your app will be live at: `https://your-project.vercel.app`

---

## 🔗 Part 3: Connect Frontend & Backend

### Update Backend with Frontend URL
1. Go back to Render dashboard
2. Open your backend service
3. Go to **Environment** tab
4. Add/Update:
   - **Key**: `FRONTEND_URL`
   - **Value**: `https://your-project.vercel.app` (your Vercel URL)
5. Save (this will trigger a redeploy)

---

## ✅ Verify Deployment

### Test Backend
Visit: `https://neuroscan-ai-backend.onrender.com/health`

Should return:
```json
{"status": "healthy"}
```

### Test Frontend
1. Visit your Vercel URL: `https://your-project.vercel.app`
2. Try uploading a test image from `test-samples/`
3. **First request may take 30-60 seconds** (Render cold start on free tier)

---

## ⚡ Important Notes

### Free Tier Limitations

**Render (Backend):**
- ⚠️ **Sleeps after 15 min inactivity**
- ⚠️ **Cold start**: 30-60 seconds on first request
- ✅ **Auto-deploys** on git push
- ✅ **HTTPS included**

**Vercel (Frontend):**
- ✅ **Always on** - instant loading
- ✅ **Global CDN**
- ✅ **Auto-deploys** on git push
- ✅ **Preview deployments** for PRs

---

## 🔄 Updating Your App

### Auto-Deployment
Both platforms auto-deploy when you push to GitHub:

```bash
git add .
git commit -m "Update feature"
git push
```

- Vercel: Redeploys in 1-2 min
- Render: Redeploys in 5-10 min

---

## 🐛 Troubleshooting

### Backend Returns 500 Error
- Check Render logs: Dashboard → Logs
- Common issue: Missing dependencies in requirements.txt

### Frontend Can't Connect to Backend
- Check CORS settings in backend
- Verify `VITE_API_URL` in Vercel environment variables
- Check browser console for errors

### Cold Start Too Slow
- First request after 15 min takes 30-60s (normal on free tier)
- Consider upgrading Render to paid tier ($7/month) for always-on

---

## 💰 Cost Breakdown

| Service | Plan | Cost | Features |
|---------|------|------|----------|
| Vercel | Free | $0 | 100GB bandwidth, unlimited projects |
| Render | Free | $0 | 750 hours/month, sleeps after inactivity |
| **Total** | | **$0/month** | Perfect for portfolio/demo |

---

## 🎉 You're Live!

Your NeuroScan AI is now deployed and accessible worldwide!

Share your links:
- **Frontend**: `https://your-project.vercel.app`
- **API Docs**: `https://neuroscan-ai-backend.onrender.com/docs`
