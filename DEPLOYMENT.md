# Guide de Déploiement - EV Optimizer App

Ce guide vous explique comment déployer votre application Streamlit en ligne.

## 🚀 Option 1 : Streamlit Community Cloud (RECOMMANDÉ - Gratuit)

La solution la plus simple pour déployer une app Streamlit.

### Prérequis
- Un compte GitHub (gratuit)
- Un compte Streamlit Cloud (gratuit - https://share.streamlit.io/)

### Étapes

1. **Utiliser votre repository existant ou créer un nouveau**
   - Votre GitHub : https://github.com/ethan-bns24
   - Vous avez déjà un repository "EV-APP" - vous pouvez l'utiliser ou créer un nouveau repository
   - Allez sur https://github.com/new pour créer un nouveau repository si nécessaire
   - **Ne pas** initialiser avec README, .gitignore, ou licence si vous créez un nouveau repo

2. **Préparer votre code localement**
   ```bash
   cd /Users/ethanbns/Documents/EV-App
   
   # Initialiser git si pas déjà fait
   git init
   
   # Le .gitignore existe déjà, vérifiez qu'il est bien présent
   
   # Ajouter tous les fichiers nécessaires
   git add app.py requirements.txt README.md DEPLOYMENT.md .gitignore
   git commit -m "Initial commit: EV Optimizer App with deployment config"
   ```

3. **Pousser vers GitHub**
   
   **Option A : Utiliser le repository EV-APP existant**
   ```bash
   git remote add origin https://github.com/ethan-bns24/EV-APP.git
   git branch -M main
   git push -u origin main
   ```
   
   **Option B : Créer un nouveau repository (recommandé pour un déploiement propre)**
   ```bash
   # Créez d'abord "ev-optimizer-app" sur GitHub, puis :
   git remote add origin https://github.com/ethan-bns24/ev-optimizer-app.git
   git branch -M main
   git push -u origin main
   ```

4. **Déployer sur Streamlit Cloud**
   - Allez sur https://share.streamlit.io/
   - Connectez-vous avec GitHub
   - Cliquez sur "New app"
   - Sélectionnez votre repository et la branche `main`
   - Le fichier principal : `app.py`
   - Cliquez sur "Deploy"

5. **Configuration des variables d'environnement (si nécessaire)**
   - Si vous avez une clé API à sécuriser, allez dans "Settings" → "Secrets"
   - Ajoutez vos clés API là (ex: `OPENROUTESERVICE_API_KEY=your_key_here`)

Votre app sera accessible à l'adresse : `https://ev-optimizer-app.streamlit.app` (ou le nom que vous avez choisi)

---

## 🌐 Option 2 : Render (Gratuit avec limites)

### Étapes

1. **Créer un compte sur Render** : https://render.com

2. **Créer un nouveau Web Service**
   - Connectez votre repository GitHub
   - Sélectionnez "Python 3" comme environnement
   - Commande de build : `pip install -r requirements.txt`
   - Commande de démarrage : `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

3. **Variables d'environnement**
   - Ajoutez vos clés API dans la section "Environment"

---

## ☁️ Option 3 : Railway (Simple et payant après crédit gratuit)

### Étapes

1. **Créer un compte Railway** : https://railway.app

2. **Nouveau projet depuis GitHub**
   - Connectez votre repository
   - Railway détecte automatiquement Python

3. **Configuration**
   - Railway détectera automatiquement Streamlit
   - Assurez-vous que `requirements.txt` est présent

4. **Variables d'environnement**
   - Ajoutez vos clés API dans les variables d'environnement

---

## 🐳 Option 4 : Docker + Hébergement (Avancé)

Si vous voulez plus de contrôle, vous pouvez utiliser Docker.

### Créer un Dockerfile

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

Ensuite, déployez sur :
- **DigitalOcean App Platform**
- **AWS App Runner**
- **Google Cloud Run**
- **Azure Container Instances**

---

## 📋 Checklist avant déploiement

- [ ] Vérifier que `requirements.txt` contient toutes les dépendances
- [ ] Tester l'app localement avec `streamlit run app.py`
- [ ] S'assurer qu'aucune clé API n'est hardcodée dans le code
- [ ] Vérifier que les fichiers sensibles sont dans `.gitignore`
- [ ] Tester avec un repository GitHub public/privé selon vos besoins

---

## 🔐 Sécurité - Variables d'environnement

### Pour Streamlit Community Cloud

Créez un fichier `.streamlit/secrets.toml` dans votre repository (ou utilisez l'interface web) :

```toml
[secrets]
OPENROUTESERVICE_API_KEY = "votre_clé_ici"
```

Puis dans votre code :
```python
import streamlit as st
import os

# Récupérer la clé depuis les secrets ou l'environnement
if "OPENROUTESERVICE_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTESERVICE_API_KEY"]
elif "OPENROUTESERVICE_API_KEY" in os.environ:
    api_key = os.environ["OPENROUTESERVICE_API_KEY"]
else:
    api_key = st.text_input("OpenRouteService API Key", type="password")
```

---

## 🎯 Recommandation

Pour un déploiement rapide et gratuit, **Streamlit Community Cloud** est la meilleure option.
C'est gratuit, facile à configurer, et parfaitement optimisé pour les apps Streamlit.

---

## 📞 Support

Si vous rencontrez des problèmes :
- Documentation Streamlit Cloud : https://docs.streamlit.io/streamlit-community-cloud
- Forum Streamlit : https://discuss.streamlit.io/

