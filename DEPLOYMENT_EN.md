# Deployment Guide - EV Optimizer App

This guide explains how to deploy your Streamlit application online.

## 🚀 Option 1: Streamlit Community Cloud (RECOMMENDED - Free)

The simplest way to deploy a Streamlit app.

### Prerequisites
- A GitHub account (free)
- A Streamlit Cloud account (free — https://share.streamlit.io/)

### Steps

1. Use your existing repository or create a new one
   - Your GitHub: https://github.com/ethan-bns24
   - You already have a repository "EV-APP" — use it or create a new repository
   - Go to https://github.com/new to create a new repository if needed
   - Do not initialize with README, .gitignore, or license if creating a new repo

2. Prepare your code locally
   ```bash
   cd /Users/ethanbns/Documents/EV-App
   
   # Initialize git if needed
   git init
   
   # Add necessary files
   git add app.py requirements.txt README_EN.md DEPLOYMENT_EN.md .gitignore
   git commit -m "Initial commit: EV Optimizer App (EN) with deployment config"
   ```

3. Push to GitHub
   
   Option A: Use the existing EV-APP repository
   ```bash
   git remote add origin https://github.com/ethan-bns24/EV-APP.git
   git branch -M main
   git push -u origin main
   ```
   
   Option B: Create a new repository (recommended for a clean deployment)
   ```bash
   # Create "ev-optimizer-app" on GitHub, then:
   git remote add origin https://github.com/ethan-bns24/ev-optimizer-app.git
   git branch -M main
   git push -u origin main
   ```

4. Deploy on Streamlit Cloud
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch `main`
   - Main file: `app.py`
   - Click "Deploy"

5. Environment variables (if needed)
   - If you need to secure your API key, go to "Settings" → "Secrets"
   - Add your API keys there (e.g., `OPENROUTESERVICE_API_KEY=your_key_here`)

Your app will be available at a URL like: `https://ev-optimizer-app.streamlit.app` (or the name you choose)

---

## 🌐 Option 2: Render (Free with limits)

### Steps

1. Create a Render account: https://render.com

2. Create a new Web Service
   - Connect your GitHub repository
   - Select "Python 3" as environment
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

3. Environment variables
   - Add your API keys in the Environment section

---

## ☁️ Option 3: Railway (Simple and paid after free credit)

### Steps

1. Create a Railway account: https://railway.app
2. New project from GitHub
   - Connect your repository
   - Railway auto-detects Python
3. Configuration
   - Railway often auto-detects Streamlit
   - Ensure `requirements.txt` is present
4. Environment variables
   - Add your API keys as environment variables

---

## 🐳 Option 4: Docker + Hosting (Advanced)

If you want more control, you can use Docker.

### Create a Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then deploy to:
- DigitalOcean App Platform
- AWS App Runner
- Google Cloud Run
- Azure Container Instances

---

## 📋 Pre-deployment checklist

- [ ] Ensure `requirements.txt` contains all dependencies
- [ ] Test locally with `streamlit run app.py`
- [ ] Ensure no API keys are hard-coded
- [ ] Ensure sensitive files are in `.gitignore`
- [ ] Test with a public/private GitHub repo as needed

---

## 🔐 Secrets and environment variables

### For Streamlit Community Cloud

Create a `.streamlit/secrets.toml` in your repository (or use the web UI):

```toml
[secrets]
OPENROUTESERVICE_API_KEY = "your_key_here"
```

Then in your code:
```python
import streamlit as st
import os

if "OPENROUTESERVICE_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTESERVICE_API_KEY"]
elif "OPENROUTESERVICE_API_KEY" in os.environ:
    api_key = os.environ["OPENROUTESERVICE_API_KEY"]
else:
    api_key = st.text_input("OpenRouteService API Key", type="password")
```

---

## 🎯 Recommendation

For a quick and free deployment, Streamlit Community Cloud is the best option.

---

## 📞 Support

If you run into issues:
- Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit forum: https://discuss.streamlit.io/

