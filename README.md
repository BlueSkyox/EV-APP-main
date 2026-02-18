# 🚗🔋 EV Optimizer - Optimiseur de Vitesse pour Véhicules Électriques

Application web Streamlit pour optimiser la consommation énergétique de votre véhicule électrique en planifiant des trajets intelligents.

## ✨ Fonctionnalités

- 📍 **Calcul d'itinéraire optimisé** avec OpenRouteService
- ⚡ **Optimisation de la consommation énergétique** selon la vitesse
- 🚦 **Limitations de vitesse par segment** (autoroute, route nationale, ville)
- 🛣️ **Détection intelligente des carrefours** et points de ralentissement
- 📊 **Graphiques détaillés** de consommation vs vitesse
- 🔋 **Planification des recharges** avec calcul du nombre d'arrêts nécessaires
- 👥 **Prise en compte du poids** (véhicule + passagers)
- 🌡️ **Gestion de la climatisation** dans les calculs
- ⛰️ **Prise en compte du relief** et des dénivelés

## 🚀 Démarrage rapide

### Installation locale

1. **Cloner le repository**
```bash
git clone https://github.com/ethan-bns24/EV-APP.git
cd EV-APP
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer l'application**
```bash
streamlit run app.py
```

5. **Ouvrir dans le navigateur**
   - L'application s'ouvrira automatiquement sur http://localhost:8501

## 🔑 Configuration

### Clé API OpenRouteService

1. Créez un compte gratuit sur https://openrouteservice.org/
2. Générez une clé API
3. Entrez la clé dans l'interface de l'application (sidebar)

## 📦 Structure du projet

```
EV-App/
├── app.py              # Application principale
├── requirements.txt    # Dépendances Python
├── .gitignore         # Fichiers à ignorer dans Git
├── DEPLOYMENT.md      # Guide de déploiement
└── README.md          # Ce fichier
```

## 🌐 Déploiement en ligne

Voir le guide complet dans [DEPLOYMENT.md](DEPLOYMENT.md)

**Option rapide : Streamlit Community Cloud**
1. Poussez votre code sur GitHub
2. Allez sur https://share.streamlit.io/
3. Connectez votre repository
4. Déployez en un clic !

## 📊 Modèles de véhicules supportés

- Tesla Model 3, Model Y
- Audi Q4 e-tron
- BMW iX3, i3
- Mercedes EQC
- Volkswagen ID.4
- Renault Zoe
- Nissan Leaf
- Hyundai IONIQ 5
- Kia EV6
- Profil personnalisé

## 🛠️ Technologies utilisées

- **Streamlit** : Framework web Python
- **OpenRouteService** : API de routage et géocodage
- **Matplotlib** : Visualisation de données
- **Pandas** : Manipulation de données
- **NumPy** : Calculs scientifiques

## 📝 Licence

Ce projet est libre d'utilisation pour des fins éducatives et personnelles.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📧 Contact

Pour toute question ou suggestion, ouvrez une issue sur GitHub.

---

Fait avec ❤️ pour optimiser votre expérience de conduite électrique

