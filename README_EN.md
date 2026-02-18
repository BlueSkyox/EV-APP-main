# 🚗🔋 EV Optimizer — EV Eco‑Speed Advisory App

Streamlit web application to optimize the energy consumption of your electric vehicle by planning intelligent trips.

## ✨ Features

- 📍 Optimized routing with OpenRouteService
- ⚡ Energy consumption optimization as a function of speed
- 🚦 Per‑segment speed limits (motorway, primary road, city)
- 🛣️ Smart detection of intersections and slowdown points
- 📊 Detailed charts (energy vs speed, time vs speed)
- 🔋 Charging planning with estimated number of stops
- 👥 Passenger weight consideration (vehicle + passengers)
- 🌡️ HVAC consideration in calculations
- ⛰️ Relief and elevation profile taken into account

## 🚀 Quickstart

### Local install

1. Clone the repository
```bash
git clone https://github.com/ethan-bns24/EV-APP.git
cd EV-APP
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app.py
```

5. Open in your browser
- The app will open automatically at http://localhost:8501

## 🔑 Configuration

### OpenRouteService API Key

1. Create a free account at https://openrouteservice.org/
2. Generate an API key
3. Paste the key in the app sidebar

## 📦 Project structure

```
EV-App/
├── app.py              # Main app
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore list
├── DEPLOYMENT.md       # Deployment guide (FR)
├── DEPLOYMENT_EN.md    # Deployment guide (EN)
├── README.md           # README (FR)
└── README_EN.md        # README (EN)
```

## 🌐 Online deployment

See the full guide in `DEPLOYMENT_EN.md`.

### Quick option: Streamlit Community Cloud
1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repository
4. One‑click deploy!

## 📊 Supported vehicle models

- Tesla Model 3, Model Y
- Audi Q4 e‑tron
- BMW iX3, i3
- Mercedes EQC
- Volkswagen ID.4
- Renault Zoe
- Nissan Leaf
- Hyundai IONIQ 5
- Kia EV6
- Custom profile

## 🛠️ Tech stack

- Streamlit (web framework)
- OpenRouteService (routing & geocoding API)
- Matplotlib (visualization)
- Pandas (data manipulation)
- NumPy (scientific computing)

## 📝 License

This project is free to use for educational and personal purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or a pull request.

## 📧 Contact

For questions or suggestions, open an issue on GitHub.

---

Made with ❤️ to optimize your EV driving experience.


