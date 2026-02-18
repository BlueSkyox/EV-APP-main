# 🚀 Déploiement Rapide - Guide Express

## Pour ethan-bns24 (https://github.com/ethan-bns24)

### Étape 1 : Préparer le code localement

```bash
cd /Users/ethanbns/Documents/EV-App

# Vérifier l'état git
git status

# Si pas de repo git initialisé :
git init

# Ajouter tous les fichiers nécessaires
git add app.py requirements.txt README.md DEPLOYMENT.md .gitignore

# Commit
git commit -m "Deployment ready: EV Optimizer App"

# Vérifier que vous êtes bien connecté à GitHub
git remote -v
```

### Étape 2 : Connecter à GitHub

**Option 1 : Utiliser le repository EV-APP existant**
```bash
git remote add origin https://github.com/ethan-bns24/EV-APP.git
git branch -M main
git push -u origin main
```

**Option 2 : Créer un nouveau repository dédié (RECOMMANDÉ)**
1. Allez sur https://github.com/new
2. Nom : `ev-optimizer-app` ou `ev-speed-optimizer`
3. Description : "Optimiseur de vitesse pour véhicules électriques"
4. **Public** (pour Streamlit Cloud gratuit)
5. Ne cochez PAS "Add README", "Add .gitignore", "Choose a license"
6. Cliquez sur "Create repository"
7. Puis :
```bash
git remote add origin https://github.com/ethan-bns24/ev-optimizer-app.git
git branch -M main
git push -u origin main
```

### Étape 3 : Déployer sur Streamlit Cloud

1. **Aller sur Streamlit Cloud**
   - https://share.streamlit.io/
   - Connectez-vous avec votre compte GitHub (ethan-bns24)

2. **Créer une nouvelle app**
   - Cliquez sur "New app"
   - **Repository** : Sélectionnez `ethan-bns24/EV-APP` ou votre nouveau repo
   - **Branch** : `main`
   - **Main file path** : `app.py`
   - **App URL** (optionnel) : Personnalisez l'URL finale

3. **Configurer les secrets (optionnel)**
   - Si vous voulez cacher la clé API, allez dans "Settings" → "Secrets"
   - Ajoutez :
     ```toml
     [secrets]
     OPENROUTESERVICE_API_KEY = "votre_clé_ici"
     ```
   - Puis modifiez votre code pour utiliser `st.secrets["OPENROUTESERVICE_API_KEY"]`

4. **Cliquez sur "Deploy"**

5. **Attendre 1-2 minutes** - Streamlit va :
   - Installer les dépendances depuis `requirements.txt`
   - Lancer votre application
   - Vous donner une URL publique

### Étape 4 : Votre app est en ligne ! 🎉

Votre URL sera du type : `https://ev-optimizer-app.streamlit.app`

## 🔄 Mises à jour futures

Pour mettre à jour votre app en ligne :
```bash
git add .
git commit -m "Description des modifications"
git push origin main
```

Streamlit Cloud redéploiera automatiquement !

## ⚠️ Troubleshooting

**Erreur "requirements.txt not found"**
- Vérifiez que `requirements.txt` est bien dans le repository

**Erreur d'import**
- Vérifiez que toutes les dépendances sont dans `requirements.txt`
- Testez localement avec `streamlit run app.py` avant de déployer

**App ne se charge pas**
- Vérifiez les logs dans Streamlit Cloud (onglet "Logs")
- Assurez-vous que `app.py` est le fichier principal

---

**Besoin d'aide ?** Voir le guide complet dans `DEPLOYMENT.md`


