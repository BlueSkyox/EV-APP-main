
EV Eco‑Speed Advisory App — Prototype (Streamlit)
=================================================

Ce prototype calcule une vitesse de croisière "éco" pour un trajet donné,
en estimant énergie et temps pour plusieurs vitesses candidates puis en
choisissant la meilleure sous contrainte d'allongement de temps.

Prérequis
---------
1. Installer Python 3.10+ (ou Anaconda).
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Obtenir une clé API **OpenRouteService** (gratuite) : https://openrouteservice.org/dev/#/signup

Lancement
---------
Depuis ce dossier :
```bash
streamlit run app.py
```
Ouvrez le lien local affiché par Streamlit (généralement http://localhost:8501).

Utilisation
-----------
- Saisir une **origine** et une **destination** (adresse/ville).
- Coller votre **clé API ORS** dans le panneau latéral.
- (Optionnel) Ajuster les **paramètres véhicule** et la **liste des vitesses** à tester.
- Cliquer sur **Calculer la vitesse conseillée**.

Notes techniques
----------------
- Le modèle énergétique simplifié tient compte de la traînée aérodynamique,
  de la résistance au roulement et du relief (pente). La régénération sur
  descente est modélisée via un rendement paramétrable.
- Les altitudes sont récupérées via ORS (`elevation=True` dans directions
  ou via `elevation/line`), puis la pente est estimée segment par segment.
- Le critère de sélection par défaut **minimise l'énergie** sous contrainte
  d'allongement de temps (% vs la vitesse la plus rapide parmi vos candidats).
  Vous pouvez aussi utiliser un score pondéré **E + λ·T**.

Aller plus loin
---------------
- Ajouter une carte interactive (folium + streamlit-folium).
- Intégrer les limites de vitesse par tronçon et un profil de vitesse segmenté.
- Charger des profils véhicule spécifiques (CdA, masse, pneus) par modèle de VE.

Tu penses qu'on pourrait faire des améliorations graphique du site, mettre le nombre 
de personne avec le poids, l'utilisation de la clim ect pour etre plus précis et rajouter 
une option ou l'on rentre le pourcentage de batterie au départ et avec le pourcentage à la 
fin combien de recharge on doit faire

./launch.sh