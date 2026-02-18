# 🚀 Quick Deployment Guide

## For ethan-bns24 (https://github.com/ethan-bns24)

### Step 1: Prepare code locally

```bash
cd /Users/ethanbns/Documents/EV-App

# Check git status
git status

# If repo is not initialized:
git init

# Add required files
git add app.py requirements.txt README_EN.md DEPLOYMENT_EN.md .gitignore

# Commit
git commit -m "Deployment ready: EV Optimizer App (EN)"

# Check GitHub remotes
git remote -v
```

### Step 2: Connect to GitHub

**Option 1: Use existing EV-APP repository**
```bash
git remote add origin https://github.com/ethan-bns24/EV-APP.git
git branch -M main
git push -u origin main
```

**Option 2: Create a new dedicated repository (RECOMMENDED)**
1. Go to https://github.com/new
2. Name: `ev-optimizer-app` or `ev-speed-optimizer`
3. Description: "EV Eco‑Speed Optimizer"
4. Public (for free Streamlit Cloud)
5. Do NOT check "Add README", ".gitignore", or license
6. Click "Create repository"
7. Then:
```bash
git remote add origin https://github.com/ethan-bns24/ev-optimizer-app.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to Streamlit Cloud
   - https://share.streamlit.io/
   - Sign in with GitHub (ethan-bns24)

2. Create a new app
   - Click "New app"
   - Repository: select `ethan-bns24/EV-APP` or your new repo
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (optional): customize your final URL

3. Configure secrets (optional)
   - If you want to keep the API key private, go to "Settings" → "Secrets"
   - Add:
     ```toml
     [secrets]
     OPENROUTESERVICE_API_KEY = "your_key_here"
     ```
   - Update code to use `st.secrets["OPENROUTESERVICE_API_KEY"]`

4. Click "Deploy"

5. Wait 1–2 minutes — Streamlit will:
   - Install dependencies from `requirements.txt`
   - Launch your app
   - Provide a public URL

### Step 4: Your app is online! 🎉

Your URL will look like: `https://ev-optimizer-app.streamlit.app`

## 🔄 Future updates

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

Streamlit Cloud redeploys automatically!

## ⚠️ Troubleshooting

**Error "requirements.txt not found"**
- Ensure `requirements.txt` is present in the repository

**Import error**
- Ensure all dependencies are listed in `requirements.txt`
- Test locally with `streamlit run app.py` before deployment

**App not loading**
- Check logs in Streamlit Cloud ("Logs" tab)
- Ensure `app.py` is the main file

---

See the full guide in `DEPLOYMENT_EN.md`.

