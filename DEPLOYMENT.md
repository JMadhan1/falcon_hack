# Deployment Guide for Render

## 📋 Prerequisites
All deployment files have been created for you:
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `build.sh` - Build script for system dependencies
- ✅ `Procfile` - Process file for Render

## 🚀 Deployment Steps

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Space Station Safety Detector"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Render

1. **Go to [Render Dashboard](https://dashboard.render.com/)**
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   **Basic Settings:**
   - **Name:** `space-station-detector` (or your choice)
   - **Region:** Choose closest to you
   - **Branch:** `main`
   - **Root Directory:** Leave empty
   - **Runtime:** `Python 3`

   **Build & Deploy:**
   - **Build Command:** `./build.sh`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

   **Instance Type:**
   - **Free** (for testing) or **Starter** (for better performance)

   **Environment Variables:**
   - No additional variables needed

5. Click **"Create Web Service"**

### 3. Wait for Deployment
- Render will build and deploy your app (takes 5-10 minutes)
- You'll get a URL like: `https://space-station-detector.onrender.com`

## ⚠️ Important Notes

### Model File
Your trained model (`runs/detect/train/weights/best.pt`) needs to be included:

**Option 1: Include in Git (if < 100MB)**
```bash
git add runs/detect/train/weights/best.pt
git commit -m "Add trained model"
git push
```

**Option 2: Use Git LFS (if > 100MB)**
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add runs/detect/train/weights/best.pt
git commit -m "Add model with LFS"
git push
```

**Option 3: Upload to Cloud Storage**
- Upload model to Google Drive/Dropbox
- Modify `app.py` to download model on startup
- Add download logic in `load_model()` function

### Free Tier Limitations
- **Spin down after 15 min of inactivity**
- **750 hours/month free**
- **First request may be slow** (cold start)

### Upgrade to Starter ($7/month) for:
- ✅ No spin down
- ✅ Faster performance
- ✅ More memory

## 🔧 Troubleshooting

### Build Fails
- Check `build.sh` has execute permissions
- Verify all dependencies in `requirements.txt`

### App Doesn't Start
- Check logs in Render dashboard
- Verify `Procfile` syntax
- Ensure port binding is correct

### Model Not Found
- Verify model file is in repository
- Check path in `app.py`: `runs/detect/train/weights/best.pt`

### Slow Performance
- Upgrade to Starter plan
- Optimize model (use smaller YOLO variant)
- Add caching for model loading

## 📊 Monitoring
- View logs in Render dashboard
- Monitor performance metrics
- Set up alerts for downtime

## 🎉 Success!
Once deployed, share your app URL:
`https://your-app-name.onrender.com`

---

**Need Help?**
- [Render Documentation](https://render.com/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
