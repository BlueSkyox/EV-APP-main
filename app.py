import math
import json
import time
import requests
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import folium
from streamlit_folium import st_folium
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from functools import lru_cache
import os
from datetime import datetime, timezone  # ✅ NEW

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="EV Eco-Speed Advisory App", layout="wide", page_icon="🚗")

# Clé ORS par défaut (peut être surchargée par st.secrets ou l'environnement)
DEFAULT_ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjA5MDkyNTdkYTlmNzQ5NmNhNjMxNzVjZGM1NTE0ZWYzIiwiaCI6Im11cm11cjY0In0="

CHARGING_STOP_DURATION_MIN = 20  # Durée moyenne d'une recharge (minutes)

# Helper: lire la clé uniquement depuis l'ENV (ou utiliser la valeur par défaut).
# Aucun accès à st.secrets pour éviter l'erreur "No secrets found".
def get_ors_key() -> str:
    return os.environ.get("OPENROUTESERVICE_API_KEY", DEFAULT_ORS_API_KEY)


def air_density_from_temp_c(temp_c: float, pressure_pa: float = 101325.0) -> float:
    """Approximate dry-air density from ambient temperature (ideal gas law).
    Uses Kelvin internally, so negative °C is safe: T_K = 273.15 + temp_c.
    """
    R_specific = 287.05  # J/(kg·K) for dry air
    T_k = 273.15 + float(temp_c)
    T_k = max(T_k, 1.0)
    return pressure_pa / (R_specific * T_k)


# ✅ NEW: Open-Meteo (free, no key) + rain→Crr factor
def fetch_open_meteo_weather(lat: float, lon: float):
    """
    Open-Meteo (free, no key): returns (temp_c, precip_mm_per_h)
    - temperature from current_weather
    - precipitation from hourly precipitation closest to current UTC hour
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "precipitation",
        "timezone": "UTC",
        "forecast_days": 1,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Temperature
    cw = data.get("current_weather", {}) if isinstance(data, dict) else {}
    temp_c = None
    if isinstance(cw, dict):
        temp_c = cw.get("temperature", None)

    # Precipitation (mm/h)
    precip = 0.0
    hourly = data.get("hourly", {}) if isinstance(data, dict) else {}
    times = hourly.get("time", [])
    precs = hourly.get("precipitation", [])
    if times and precs and len(times) == len(precs):
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # Open-Meteo hourly timestamps look like "YYYY-MM-DDTHH:00"
        now_prefix = now_utc.isoformat()[:13]  # "YYYY-MM-DDTHH"
        idx = None
        for i, t in enumerate(times):
            if isinstance(t, str) and t.startswith(now_prefix):
                idx = i
                break
        if idx is None:
            idx = 0
        try:
            precip = float(precs[idx] or 0.0)
        except Exception:
            precip = 0.0

    if temp_c is None:
        raise ValueError("Open-Meteo: temperature missing")

    return float(temp_c), max(0.0, float(precip))


def rolling_resistance_rain_factor(precip_mm_h: float) -> float:
    """
    Crr multiplier vs rain intensity (mm/h):
    dry 0       → 1.00
    light ≤1    → 1.05
    moderate ≤4 → 1.12
    heavy >4    → 1.20
    """
    p = max(0.0, float(precip_mm_h))
    if p <= 0.0:
        return 1.00
    elif p <= 1.0:
        return 1.05
    elif p <= 4.0:
        return 1.12
    else:
        return 1.20


def battery_capacity_factor(temp_c: float) -> float:
    """Simple EV model: usable battery capacity decreases in cold/heat.
    Returns factor in [0.75, 1.00]."""
    t = float(temp_c)
    cold_drop = 0.003 * max(0.0, 20.0 - t)
    heat_drop = 0.001 * max(0.0, t - 30.0)
    f = 1.0 - cold_drop - heat_drop
    return max(0.75, min(1.0, f))


def battery_energy_multiplier(temp_c: float) -> float:
    """Simple EV model: extra energy due to internal losses at non-ideal temps.
    Returns multiplier in [1.0, 1.35]."""
    t = float(temp_c)
    cold_penalty = 0.01 * max(0.0, 10.0 - t)
    heat_penalty = 0.003 * max(0.0, t - 35.0)
    m = 1.0 + cold_penalty + heat_penalty
    return max(1.0, min(1.35, m))


# ------------------------------
# ✅ NEW: CO2 presets (user-adjustable)
# ------------------------------
# These are *defaults/assumptions* so the user can model scenarios.
# You can replace these with your own verified values later.
GRID_INTENSITY_PRESETS_G_PER_KWH = {
    "Custom (enter manually)": None,
    "France (low-carbon mix) ~50 g/kWh": 50,
    "EU average ~250 g/kWh": 250,
    "World average ~475 g/kWh": 475,
    "Coal-heavy scenario ~800 g/kWh": 800,
}

# ------------------------------
# HVAC model (per-vehicle, deterministic)
# ------------------------------
def hvac_electric_power_kw(temp_c: float, intensity_pct: float, hvac_params: dict) -> float:
    """Return electrical HVAC power in kW."""
    inten = max(0.0, min(100.0, float(intensity_pct))) / 100.0
    if inten <= 0.0:
        return 0.0

    p = hvac_params or {}
    target = float(p.get("target_cabin_c", 21.0))
    deadband = float(p.get("deadband_c", 2.0))
    max_heat = float(p.get("max_heat_kw", 5.0))
    max_cool = float(p.get("max_cool_kw", 3.0))
    cabin_factor = float(p.get("cabin_factor", 1.0))
    heat_type = str(p.get("heat_type", "heat_pump")).lower()
    cop_heat = float(p.get("cop_heat", 2.5))
    cop_cool = float(p.get("cop_cool", 2.5))

    if temp_c < (target - deadband):
        delta = (target - deadband) - float(temp_c)
        q_th = min(max_heat, cabin_factor * 0.35 * delta)
        if heat_type == "resistive":
            p_elec = q_th
        else:
            p_elec = q_th / max(cop_heat, 1e-3)
    elif temp_c > (target + deadband):
        delta = float(temp_c) - (target + deadband)
        q_th = min(max_cool, cabin_factor * 0.25 * delta)
        p_elec = q_th / max(cop_cool, 1e-3)
    else:
        p_elec = 0.2 * cabin_factor

    return float(p_elec) * inten


# Style global des graphiques
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    pass

# Ajouter des styles CSS personnalisés améliorés (design sobre et professionnel)
st.markdown("""
<style>
    .stApp { background: #f5f7fa !important; background-attachment: fixed; }
    .main .block-container {
        background: #ffffff !important;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #e1e8ed;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricDelta"] { color: inherit !important; }
    [data-testid="stMetricContainer"] {
        background: #ffffff !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }
    [data-testid="stMetricContainer"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    [data-testid="stAlert"] { border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    .stButton > button {
        background: #2c3e50 !important;
        color: white !important;
        font-weight: 600;
        border: none !important;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        transition: all 0.3s;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.2);
    }
    .stButton > button:hover {
        background: #34495e !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(44, 62, 80, 0.3);
    }
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .logo-emoji { font-size: 4rem; margin-bottom: 0.5rem; display: block; }
    .logo-container h1 { color: #2c3e50; font-weight: 700; }
    .logo-container p { color: #7f8c8d; }
    hr { border: none; height: 1px; background: #e1e8ed; margin: 2rem 0; }
    h1, h2, h3 { color: #2c3e50; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e1e8ed; }
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #bdc3c7; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #95a5a6; }
    .dataframe { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); }
</style>
""", unsafe_allow_html=True)

# Custom logo and header
st.markdown("""
<div class="logo-container">
    <span class="logo-emoji">🚗🔋⚡</span>
    <h1 style="margin: 0; color: #2c3e50;">EV Eco-Speed Optimizer</h1>
    <p style="margin-top: 0.5rem; color: #7f8c8d; font-size: 1.1rem;">
        Plan trips intelligently by optimizing your energy consumption
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ What's new and features", expanded=False):
    st.markdown("""
        **✨ Recent improvements:**
        - 👥 Consideration of the **number of passengers and their weight**
        - 🌡️ **HVAC settings** for more accurate calculations
        - 🔋 **Charging planning**: battery percentage at departure and arrival
        - 📊 **Improved charts** with optimized data visualization
        - 🚦 **New: Speed limits by segment**: automatic consideration of limits by road type (motorway 130 km/h, primary 90 km/h, city 50 km/h, etc.)
        - 🛣️ **Improved intersection detection**: precise identification of intersections, roundabouts and slow-down points
        - 💶 **Trip cost estimate** based on electricity price and predicted energy consumption
        - 🌍 **CO₂ emissions estimate** linked to electricity generation mix (gCO₂/kWh)

        This tool helps you:
        - Maximize your EV driving range
        - Reduce your energy costs
        - Plan your charging stops
        - Get more realistic consumption estimates with speeds adapted to real limits
        """)
st.markdown("---")

# ------------------------------
# Vehicle Profiles
# ------------------------------
VEHICLE_PROFILES = {
    "Tesla Model 3": {"mass_kg": 1850, "cda": 0.58, "crr": 0.008, "eta_drive": 0.95, "regen_eff": 0.85, "aux_power_kw": 2.0, "battery_kwh": 75},
    "Tesla Model Y": {"mass_kg": 2000, "cda": 0.62, "crr": 0.008, "eta_drive": 0.95, "regen_eff": 0.85, "aux_power_kw": 2.2, "battery_kwh": 75},
    "Audi Q4 e-tron": {"mass_kg": 2100, "cda": 0.70, "crr": 0.009, "eta_drive": 0.92, "regen_eff": 0.80, "aux_power_kw": 2.5, "battery_kwh": 82},
    "BMW iX3": {"mass_kg": 2180, "cda": 0.68, "crr": 0.009, "eta_drive": 0.93, "regen_eff": 0.82, "aux_power_kw": 2.3, "battery_kwh": 80},
    "Mercedes EQC": {"mass_kg": 2425, "cda": 0.72, "crr": 0.010, "eta_drive": 0.91, "regen_eff": 0.78, "aux_power_kw": 2.8, "battery_kwh": 80},
    "Volkswagen ID.4": {"mass_kg": 2120, "cda": 0.66, "crr": 0.009, "eta_drive": 0.90, "regen_eff": 0.75, "aux_power_kw": 2.0, "battery_kwh": 77},
    "Renault Zoe": {"mass_kg": 1500, "cda": 0.65, "crr": 0.010, "eta_drive": 0.90, "regen_eff": 0.70, "aux_power_kw": 1.5, "battery_kwh": 52},
    "BMW i3": {"mass_kg": 1200, "cda": 0.50, "crr": 0.008, "eta_drive": 0.92, "regen_eff": 0.80, "aux_power_kw": 1.8, "battery_kwh": 42},
    "Nissan Leaf": {"mass_kg": 1600, "cda": 0.68, "crr": 0.010, "eta_drive": 0.88, "regen_eff": 0.75, "aux_power_kw": 1.7, "battery_kwh": 40},
    "Hyundai IONIQ 5": {"mass_kg": 1950, "cda": 0.64, "crr": 0.008, "eta_drive": 0.94, "regen_eff": 0.83, "aux_power_kw": 2.1, "battery_kwh": 73},
    "Kia EV6": {"mass_kg": 1980, "cda": 0.63, "crr": 0.008, "eta_drive": 0.94, "regen_eff": 0.83, "aux_power_kw": 2.1, "battery_kwh": 77},
    "Personnalisé": {"mass_kg": 1900, "cda": 0.62, "crr": 0.010, "eta_drive": 0.90, "regen_eff": 0.60, "aux_power_kw": 2.0, "battery_kwh": 60},
}

# ------------------------------
# HVAC parameters by vehicle (edit / calibrate here)
# ------------------------------
HVAC_PROFILES = {
    "Tesla Model 3": {"heat_type": "heat_pump", "max_heat_kw": 5.0, "max_cool_kw": 3.0, "cabin_factor": 1.00, "cop_heat": 2.8, "cop_cool": 2.6},
    "Tesla Model Y": {"heat_type": "heat_pump", "max_heat_kw": 5.5, "max_cool_kw": 3.2, "cabin_factor": 1.15, "cop_heat": 2.8, "cop_cool": 2.6},
    "Audi Q4 e-tron": {"heat_type": "heat_pump", "max_heat_kw": 5.5, "max_cool_kw": 3.2, "cabin_factor": 1.15, "cop_heat": 2.6, "cop_cool": 2.5},
    "BMW iX3": {"heat_type": "heat_pump", "max_heat_kw": 5.8, "max_cool_kw": 3.3, "cabin_factor": 1.18, "cop_heat": 2.6, "cop_cool": 2.5},
    "Mercedes EQC": {"heat_type": "heat_pump", "max_heat_kw": 6.0, "max_cool_kw": 3.5, "cabin_factor": 1.25, "cop_heat": 2.4, "cop_cool": 2.4},
    "Volkswagen ID.4": {"heat_type": "heat_pump", "max_heat_kw": 5.5, "max_cool_kw": 3.2, "cabin_factor": 1.15, "cop_heat": 2.5, "cop_cool": 2.4},
    "Renault Zoe": {"heat_type": "heat_pump", "max_heat_kw": 4.0, "max_cool_kw": 2.5, "cabin_factor": 0.85, "cop_heat": 2.4, "cop_cool": 2.4},
    "BMW i3": {"heat_type": "resistive", "max_heat_kw": 4.0, "max_cool_kw": 2.2, "cabin_factor": 0.80, "cop_heat": 1.0, "cop_cool": 2.3},
    "Nissan Leaf": {"heat_type": "resistive", "max_heat_kw": 4.5, "max_cool_kw": 2.5, "cabin_factor": 0.90, "cop_heat": 1.0, "cop_cool": 2.3},
    "Hyundai IONIQ 5": {"heat_type": "heat_pump", "max_heat_kw": 5.5, "max_cool_kw": 3.2, "cabin_factor": 1.10, "cop_heat": 2.6, "cop_cool": 2.5},
    "Kia EV6": {"heat_type": "heat_pump", "max_heat_kw": 5.5, "max_cool_kw": 3.2, "cabin_factor": 1.10, "cop_heat": 2.6, "cop_cool": 2.5},
    "Personnalisé": {"heat_type": "heat_pump", "max_heat_kw": 5.0, "max_cool_kw": 3.0, "cabin_factor": 1.00, "cop_heat": 2.5, "cop_cool": 2.5},
}

# ------------------------------
# Sidebar – Parameters
# ------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚗⚡</div>
        <div style="font-weight: 600; color: #667eea; font-size: 1.1rem;">EV Optimizer</div>
    </div>
    """, unsafe_allow_html=True)

    st.header("⚙️ Settings")

    with st.expander("💡 Tip", expanded=False):
        st.markdown("""
        **Factors that affect your consumption:**
        - 🧍 **Weight**: More passengers = higher consumption
        - 🌡️ **HVAC**: Can increase consumption by 10–30%
        - 🏔️ **Topography**: Uphill segments significantly increase consumption
        - 🏎️ **Speed**: Consumption rises exponentially with speed
        - 🌡️ **Temperature**: Extreme cold/heat reduces battery efficiency
        """)

    ors_key = get_ors_key()
    st.caption("Using configured OpenRouteService API key")
    st.markdown("---")

    st.subheader("🚗 Vehicle profile")
    vehicle_profile = st.selectbox("Model", list(VEHICLE_PROFILES.keys()))

    if vehicle_profile != "Personnalisé":
        profile = VEHICLE_PROFILES[vehicle_profile]
        st.info(f"Profile {vehicle_profile} loaded")
        st.caption(f"Battery: {profile['battery_kwh']} kWh | Aux: {profile['aux_power_kw']} kW")

    hvac_params = HVAC_PROFILES.get(vehicle_profile, HVAC_PROFILES.get("Personnalisé", {})).copy()

    st.markdown("---")
    st.subheader("Vehicle parameters")

    if vehicle_profile != "Personnalisé":
        profile = VEHICLE_PROFILES[vehicle_profile]
        mass_kg = st.number_input("Mass (kg)", 1000, 3500, profile["mass_kg"], 50, disabled=True)
        cda = st.number_input("Frontal area × Cd (CdA in m²)", 0.3, 1.2, profile["cda"], 0.01, disabled=True)
        crr = st.number_input("Rolling resistance (Crr)", 0.005, 0.02, profile["crr"], 0.001, format="%.3f", disabled=True)
        eta_drive = st.slider("Drivetrain efficiency (η)", 0.70, 0.98, profile["eta_drive"], 0.01, disabled=True)
        regen_eff = st.slider("Regeneration efficiency (%)", 0, 90, int(profile["regen_eff"]*100), 5, disabled=True) / 100.0
        aux_power_kw = st.number_input("Auxiliary power (kW)", 0.0, 5.0, profile["aux_power_kw"], 0.1, disabled=True)
        battery_kwh = st.number_input("Battery capacity (kWh)", 20, 150, profile["battery_kwh"], 5, disabled=True)
    else:
        mass_kg = st.number_input("Mass (kg)", 1000, 3500, 1900, 50)
        cda = st.number_input("Frontal area × Cd (CdA in m²)", 0.3, 1.2, 0.62, 0.01)
        crr = st.number_input("Rolling resistance (Crr)", 0.005, 0.02, 0.010, 0.001, format="%.3f")
        eta_drive = st.slider("Drivetrain efficiency (η)", 0.70, 0.98, 0.90, 0.01)
        regen_eff = st.slider("Regeneration efficiency (%)", 0, 90, 60, 5) / 100.0
        aux_power_kw = st.number_input("Auxiliary power (kW)", 0.0, 5.0, 2.0, 0.1)
        battery_kwh = st.number_input("Battery capacity (kWh)", 20, 150, 60, 5)

        st.markdown("---")
        st.subheader("HVAC parameters (custom)")
        hvac_params["heat_type"] = st.selectbox("Heating system", ["heat_pump", "resistive"], index=0)
        hvac_params["cabin_factor"] = st.slider("Cabin factor", 0.6, 1.6, float(hvac_params.get("cabin_factor", 1.0)), 0.05)
        hvac_params["max_heat_kw"] = st.slider("Max heating (kW_th)", 2.0, 8.0, float(hvac_params.get("max_heat_kw", 5.0)), 0.1)
        hvac_params["max_cool_kw"] = st.slider("Max cooling (kW_th)", 1.5, 6.0, float(hvac_params.get("max_cool_kw", 3.0)), 0.1)
        hvac_params["cop_heat"] = st.slider("COP heating (heat pump)", 1.0, 4.0, float(hvac_params.get("cop_heat", 2.5)), 0.1)
        hvac_params["cop_cool"] = st.slider("COP cooling", 1.0, 4.0, float(hvac_params.get("cop_cool", 2.5)), 0.1)
        hvac_params["target_cabin_c"] = st.slider("Target cabin temp (°C)", 18, 24, int(hvac_params.get("target_cabin_c", 21)), 1)
        hvac_params["deadband_c"] = st.slider("Deadband (°C)", 0.5, 3.0, float(hvac_params.get("deadband_c", 2.0)), 0.5)

    # ✅ UPDATED Weather section (Open-Meteo option)
    st.subheader("🌦️ Weather")
    use_live_weather = st.checkbox("Use live weather (Open-Meteo)", value=False)

    temp_c_manual = st.slider("Ambient temperature (°C)", -20, 45, 15, 1)
    precip_mm_h_manual = st.slider("Rain intensity (mm/h) [manual]", 0.0, 20.0, 0.0, 0.5)

    # Default values (can be overridden on run)
    temp_c = temp_c_manual
    precip_mm_h = precip_mm_h_manual

    rho_air = air_density_from_temp_c(temp_c)
    rain_factor_preview = rolling_resistance_rain_factor(precip_mm_h)
    st.caption(f"Air density: {rho_air:.3f} kg/m³ | Rain: {precip_mm_h:.1f} mm/h → Crr×{rain_factor_preview:.2f}")

    st.subheader("🔋 Battery temperature effects")
    use_battery_temp = st.checkbox("Model battery temp impact (EV)", value=True)
    batt_cap_fac = battery_capacity_factor(temp_c) if use_battery_temp else 1.0
    batt_E_mult = battery_energy_multiplier(temp_c) if use_battery_temp else 1.0
    st.caption(f"Usable capacity factor: {batt_cap_fac:.2f} | Energy multiplier: {batt_E_mult:.2f}")

    st.markdown("---")
    st.subheader("Candidate speeds (km/h)")
    default_speeds = list(range(50, 131, 5))
    speeds_str = st.text_input("Comma-separated list", ", ".join(map(str, default_speeds)))
    try:
        candidate_speeds = sorted({int(s.strip()) for s in speeds_str.split(",") if s.strip()})
    except Exception:
        candidate_speeds = default_speeds

    user_speed_limit = st.number_input("Max speed on route (km/h)", 50, 130, 110, 10, help="To stay realistic if the API does not return a limit.")
    st.markdown("---")
    st.subheader("Constraints / Objective")
    max_time_penalty_pct = st.slider("Max time increase vs fastest speed (%)", 0, 50, 15, 1)
    minimize_target = st.selectbox("Objective", ["Minimize energy under time constraint", "Weighted score (E + λ·T)"])
    lam = st.slider("λ (time weight) [for weighted score]", 0.0, 10.0, 2.0, 0.5)

    st.markdown("---")
    st.subheader("👥 Load and passengers")
    num_passengers = st.number_input("Number of passengers", 0, 7, 1, 1, help="Driver included")
    avg_weight_kg = st.number_input("Average weight per person (kg)", 40, 120, 75, 5)
    total_passenger_weight = num_passengers * avg_weight_kg

    st.markdown("---")
    st.subheader("🌡️ Driving conditions")
    use_climate = st.checkbox("Use HVAC", value=False)
    if use_climate:
        climate_intensity = st.slider("HVAC intensity (%)", 0, 100, 50, 10)
    else:
        climate_intensity = 0

    st.markdown("---")
    st.subheader("🔋 Battery state")
    battery_start_pct = st.slider("Battery at departure (%)", 20, 100, 100, 5)
    battery_end_pct = st.slider("Target battery on arrival (%)", 5, 90, 20, 5)

    # ------------------------------
    # ✅ NEW: Cost + CO2 controls
    # ------------------------------
    st.markdown("---")
    st.subheader("💶 Cost & 🌍 CO₂")
    energy_cost_per_kwh = st.number_input("Electricity cost (€/kWh)", 0.05, 1.50, 0.20, 0.01)

    grid_choice = st.selectbox("Grid CO₂ intensity preset", list(GRID_INTENSITY_PRESETS_G_PER_KWH.keys()))
    preset_val = GRID_INTENSITY_PRESETS_G_PER_KWH[grid_choice]
    if preset_val is None:
        grid_co2_g_per_kwh = st.number_input("Grid CO₂ intensity (gCO₂/kWh)", 0, 1200, 250, 10)
    else:
        grid_co2_g_per_kwh = int(preset_val)
        st.caption(f"Using preset: {grid_co2_g_per_kwh} gCO₂/kWh")

    st.markdown("---")
    st.subheader("Advanced options")
    use_elevation = st.checkbox("Use elevation data", value=True, help="Disable if you encounter API errors")
    use_segmented_speeds = st.checkbox("Speed limits by segment", value=True, help="Takes into account limits by road type (motorway, city, etc.)")
    if use_segmented_speeds:
        min_speed_delta = st.number_input("Minimum speed delta (km/h)", 0, 50, 20, 5, help="Minimum speed = limit - delta (e.g., motorway 130 with delta 20 → min 110 km/h)")
    else:
        min_speed_delta = 0
    use_detailed_route = st.checkbox("Detailed route", value=True, help="Disable for short trips")
    debug_mode = st.checkbox("Debug mode", value=False, help="Display intermediate information")

# ------------------------------
# Helpers – Physics & Energy
# ------------------------------
g = 9.81

def _haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    import math as _m
    dlat = _m.radians(lat2 - lat1)
    dlon = _m.radians(lon2 - lon1)
    a = _m.sin(dlat/2)**2 + _m.cos(_m.radians(lat1))*_m.cos(_m.radians(lat2))*_m.sin(dlon/2)**2
    c = 2 * _m.atan2(_m.sqrt(a), _m.sqrt(1-a))
    return R * c

def is_valid_ors_key(key: str) -> bool:
    if not isinstance(key, str):
        return False
    k = key.strip()
    if not k:
        return False
    banned_substrings = ["http", "Client Error", "Bad Request", "Forbidden", "Erreur", "Error:"]
    if any(b.lower() in k.lower() for b in banned_substrings):
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-.")
    if not all(c in allowed for c in k):
        return False
    return 20 <= len(k) <= 256

def calculate_charging_stops(battery_kwh, energy_needed_kwh, start_pct, end_pct):
    usable_start_kwh = battery_kwh * (start_pct / 100.0)
    target_end_kwh = battery_kwh * (end_pct / 100.0)
    safety_margin = battery_kwh * 0.10
    usable_battery = battery_kwh - safety_margin

    if usable_battery <= 0:
        return {"num_stops": 999, "usable_battery": 0, "energy_per_leg": usable_battery}

    energy_available = usable_start_kwh - max(safety_margin, target_end_kwh)

    if energy_needed_kwh <= energy_available:
        return {"num_stops": 0, "usable_battery": usable_battery, "energy_per_leg": usable_battery}

    remaining_energy = energy_needed_kwh - energy_available
    num_stops = math.ceil(remaining_energy / usable_battery)

    return {"num_stops": max(0, num_stops), "usable_battery": usable_battery, "energy_per_leg": usable_battery}

def seg_energy_and_time(distance_m, slope, speed_kmh, mass_kg, cda, crr, rho_air, eta_drive, regen_eff, aux_power_kw=0, **kwargs):
    if distance_m <= 0 or speed_kmh <= 0:
        return 0.0, 0.0

    slope = max(-0.5, min(0.5, slope))
    v = max(speed_kmh, 1e-3) * (1000/3600)

    cda_eff = float(cda)

    F_aero = 0.5 * rho_air * cda_eff * v * v
    F_roll = crr * mass_kg * g * math.cos(math.atan(slope))
    F_grade = mass_kg * g * math.sin(math.atan(slope))

    P_wheels = (F_aero + F_roll + F_grade) * v

    if P_wheels >= 0:
        P_elec = P_wheels / max(eta_drive, 1e-6)
    else:
        P_elec = P_wheels * regen_eff

    P_aux = aux_power_kw * 1000
    P_total = P_elec + P_aux

    t = distance_m / max(v, 1e-6)
    E_Wh = P_total * (t / 3600.0)

    batt_E_mult = float(kwargs.get('batt_E_mult', 1.0))
    E_Wh *= batt_E_mult
    return E_Wh, t / 3600.0

def route_energy_time(coords, elevations, speed_kmh, **veh):
    total_E = 0.0
    total_T = 0.0
    total_D = 0.0

    is_speed_list = isinstance(speed_kmh, list)
    if is_speed_list and len(speed_kmh) != len(coords) - 1:
        speed_kmh = speed_kmh[0] if speed_kmh else 50
        is_speed_list = False

    for i in range(1, len(coords)):
        if len(coords[i-1]) >= 2:
            lon1, lat1 = coords[i-1][0], coords[i-1][1]
        else:
            continue

        if len(coords[i]) >= 2:
            lon2, lat2 = coords[i][0], coords[i][1]
        else:
            continue

        R = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c
        if d < 1e-2:
            continue

        h1 = elevations[i-1]
        h2 = elevations[i]
        slope = (h2 - h1) / max(d, 1e-6)

        seg_speed = speed_kmh[i-1] if is_speed_list else speed_kmh

        Eseg, Tseg = seg_energy_and_time(d, slope, seg_speed, **veh)
        total_E += Eseg
        total_T += Tseg
        total_D += d

    return total_E, total_T, total_D / 1000.0

# ------------------------------
# Route segmentation and speed limits
# ------------------------------
def get_speed_limit_by_road_type(road_type: str, user_max_speed: int = 130) -> int:
    speed_mapping = {
        "motorway": min(130, user_max_speed),
        "trunk": min(110, user_max_speed),
        "primary": min(90, user_max_speed),
        "secondary": min(90, user_max_speed),
        "tertiary": min(90, user_max_speed),
        "unclassified": 50,
        "residential": 50,
        "service": 30,
    }
    for key, speed in speed_mapping.items():
        if road_type and key in road_type.lower():
            return speed
    return 50

def detect_intersections_improved(steps, detailed_segments, coords):
    intersections = []
    slowdown_points = []
    if not steps:
        return {"intersections": intersections, "slowdown_points": slowdown_points}

    intersection_keywords = [
        "tournez", "tourner", "turn", "tourné", "tournant",
        "roundabout", "rond-point", "rond point", "round-about",
        "bifurquez", "bifurcation", "fork", "bifurquer",
        "u-turn", "demi-tour", "uturn",
        "merge", "mergez", "fusion",
        "jonction", "junction", "join",
        "quittez", "exit", "sortie",
        "continuez", "continue",
        "prenez", "take",
        "intersection", "croisement", "crossing"
    ]

    for i, step in enumerate(steps):
        instr = str(step.get("instruction", "")).lower()
        step_type = step.get("type", 0)
        distance = step.get("distance", 0)

        if any(keyword in instr for keyword in intersection_keywords):
            intersections.append(i)

        if "roundabout" in instr or "rond-point" in instr or "rond point" in instr:
            slowdown_points.append({"type": "roundabout", "step_index": i})

        if step_type in [1, 2, 3, 4, 5, 6]:
            if distance < 100:
                slowdown_points.append({"type": "sharp_turn", "step_index": i})

    return {"intersections": intersections, "slowdown_points": slowdown_points}

def create_segmented_speeds(coords, steps, detailed_segments, candidate_speed: int, user_max_speed: int, min_speed_delta: int = 20):
    if not steps or not detailed_segments:
        return [candidate_speed] * (len(coords) - 1)

    segmented_speeds = []
    total_route_distance = sum(seg.get("distance", 0) for seg in detailed_segments) or 1

    segment_boundaries = []
    cumul_dist = 0
    for seg in detailed_segments:
        seg_dist = seg.get("distance", 0)
        segment_boundaries.append({"start": cumul_dist, "end": cumul_dist + seg_dist, "segment": seg})
        cumul_dist += seg_dist

    cumul_coord_dist = 0
    coord_distances = [0]
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i+1][0], coords[i+1][1]
        R = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        segment_distance = R * c
        cumul_coord_dist += segment_distance
        coord_distances.append(cumul_coord_dist)

    if coord_distances[-1] > 0:
        coord_ratio = total_route_distance / coord_distances[-1]
        coord_distances = [d * coord_ratio for d in coord_distances]

    found_road_types = False

    for i in range(len(coords) - 1):
        mid_dist = (coord_distances[i] + coord_distances[i+1]) / 2

        segment_found = None
        for seg_bound in segment_boundaries:
            if seg_bound["start"] <= mid_dist < seg_bound["end"]:
                segment_found = seg_bound["segment"]
                break

        if segment_found:
            steps_in_seg = segment_found.get("steps", [])
            way_type = None
            road_type = None

            if steps_in_seg:
                first_step = steps_in_seg[0]
                way_type = first_step.get("way_type", None)
                road_type = first_step.get("road_type", None)

            if not road_type and not way_type:
                way_type = segment_found.get("way_type", None)
                road_type = segment_found.get("road_type", None)

            if road_type:
                speed_limit = get_speed_limit_by_road_type(str(road_type), user_max_speed)
                found_road_types = True
                min_allowed_speed = max(30, speed_limit - min_speed_delta)
                final_speed = max(min_allowed_speed, min(candidate_speed, speed_limit))
            elif way_type:
                speed_limit = get_speed_limit_by_road_type(str(way_type), user_max_speed)
                found_road_types = True
                min_allowed_speed = max(30, speed_limit - min_speed_delta)
                final_speed = max(min_allowed_speed, min(candidate_speed, speed_limit))
            else:
                final_speed = candidate_speed

            segmented_speeds.append(final_speed)
        else:
            segmented_speeds.append(candidate_speed)

    if not found_road_types:
        return [candidate_speed] * (len(coords) - 1)

    return segmented_speeds if segmented_speeds else [candidate_speed] * (len(coords) - 1)

# ------------------------------
# OpenRouteService API wrappers
# ------------------------------
CITY_COORDS = {
    "paris, france": [2.3522, 48.8566],
    "lyon, france": [4.8357, 45.7640],
    "marseille, france": [5.3698, 43.2965],
    "beauvais, france": [2.0833, 49.4333],
    "toulouse, france": [1.4442, 43.6047],
    "nantes, france": [-1.5536, 47.2184],
}

def ors_geocode(text, api_key):
    url = "https://api.openrouteservice.org/geocode/search"
    params = {"api_key": api_key, "text": text, "size": 1}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        if not feats:
            raise ValueError("no_features")
        lon, lat = feats[0]["geometry"]["coordinates"]
        return [lon, lat]
    except Exception:
        key = (text or "").strip().lower()
        if key in CITY_COORDS:
            return CITY_COORDS[key]
        if key.endswith(", france") and key.replace(", france", "") in CITY_COORDS:
            return CITY_COORDS[key.replace(", france", "")]
        return None

def osrm_route(start_lonlat, end_lonlat, include_instructions=False):
    try:
        base = "https://router.project-osrm.org/route/v1/driving/"
        coords = f"{start_lonlat[0]},{start_lonlat[1]};{end_lonlat[0]},{end_lonlat[1]}"
        params = {"overview": "full", "geometries": "geojson", "steps": str(include_instructions).lower(), "alternatives": "false"}
        r = requests.get(base + coords, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        routes = j.get("routes", [])
        if not routes:
            return [], 0, 0
        route = routes[0]
        geometry = route.get("geometry", {})
        coords_list = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        length_m = float(route.get("distance", 0.0))
        duration_s = float(route.get("duration", 0.0))
        return coords_list, length_m, duration_s
    except Exception:
        return [], 0, 0

def osrm_route_steps(start_lonlat, end_lonlat):
    try:
        base = "https://router.project-osrm.org/route/v1/driving/"
        coords = f"{start_lonlat[0]},{start_lonlat[1]};{end_lonlat[0]},{end_lonlat[1]}"
        params = {"overview": "full", "geometries": "geojson", "steps": "true", "alternatives": "false"}
        r = requests.get(base + coords, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        routes = j.get("routes", [])
        if not routes:
            return [], []
        legs = routes[0].get("legs", [])
        steps = []
        detailed_segments = []
        for leg in legs:
            s = leg.get("steps", [])
            steps.extend(s)
            detailed_segments.append({"distance": leg.get("distance", 0), "steps": s})
        return steps, detailed_segments
    except Exception:
        return [], []

def ors_route(start_lonlat, end_lonlat, api_key, include_instructions=False):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": api_key, "Content-Type": "application/json; charset=utf-8"}
    body = {"coordinates": [start_lonlat, end_lonlat], "elevation": False, "instructions": include_instructions}
    try:
        r = requests.post(url, headers=headers, params={"format": "geojson"}, data=json.dumps(body), timeout=60)
        r.raise_for_status()
        data = r.json()
        coords = []
        length_m = 0
        duration_s = 0
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            geometry = route.get("geometry")
            if isinstance(geometry, dict) and geometry.get("type") == "LineString":
                coords = geometry.get("coordinates", [])
            summary = route.get("summary", {})
            length_m = summary.get("distance", 0)
            duration_s = summary.get("duration", 0)
        elif "features" in data and data["features"]:
            feature = data["features"][0]
            geometry = feature.get("geometry", {})
            if isinstance(geometry, dict) and geometry.get("type") == "LineString":
                coords = geometry.get("coordinates", [])
            props = feature.get("properties", {})
            if isinstance(props, dict):
                segments = props.get("segments", [])
                if segments:
                    summary = segments[0].get("summary", {})
                    length_m = summary.get("distance", 0)
                    duration_s = summary.get("duration", 0)
        if not coords:
            st.warning("Géométrie non disponible, fallback sur départ/arrivée")
            coords = [start_lonlat, end_lonlat]
        return coords, length_m, duration_s
    except Exception:
        coords, length_m, duration_s = osrm_route(start_lonlat, end_lonlat, include_instructions)
        if coords:
            st.info("Using OSRM fallback for directions")
            return coords, length_m, duration_s
        st.warning("ORS indisponible – fallback simplifié (ligne droite)")
        return [start_lonlat, end_lonlat], 0, 0

def ors_route_steps(start_lonlat, end_lonlat, api_key):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": api_key, "Content-Type": "application/json; charset=utf-8"}
    body = {"coordinates": [start_lonlat, end_lonlat], "elevation": False, "instructions": True}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
        r.raise_for_status()
        data = r.json()
        if "routes" not in data or not data["routes"]:
            raise ValueError("no_routes")
        route = data["routes"][0]
        segments = route.get("segments", [])
        steps = []
        detailed_segments = []
        for seg in segments:
            steps.extend(seg.get("steps", []))
            detailed_segments.append(seg)
        return steps, detailed_segments
    except Exception:
        steps, detailed_segments = osrm_route_steps(start_lonlat, end_lonlat)
        if steps:
            st.info("Using OSRM fallback for steps")
        return steps, detailed_segments

def ors_elevation_along(coords, api_key):
    if coords and len(coords[0]) == 3:
        return [c[2] for c in coords]

    url = "https://api.openrouteservice.org/elevation/line"
    headers = {"Authorization": api_key, "Content-Type": "application/json; charset=utf-8"}
    max_pts = 1000
    if len(coords) > max_pts:
        step = max(1, len(coords) // max_pts + (1 if len(coords) % max_pts else 0))
        reduced = coords[::step]
        if reduced[-1] != coords[-1]:
            reduced.append(coords[-1])
    else:
        reduced = coords

    body = {"format_in": "geojson", "format_out": "json", "geometry": {"type": "LineString", "coordinates": reduced}}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
        r.raise_for_status()
        data = r.json()
        elev = [pt[2] for pt in data["geometry"]["coordinates"]]

        def _interp_back(sampled, full_len):
            if len(sampled) == full_len:
                return sampled
            import numpy as np
            x_old = np.linspace(0.0, 1.0, len(sampled))
            x_new = np.linspace(0.0, 1.0, full_len)
            return np.interp(x_new, x_old, sampled).tolist()

        if len(reduced) != len(coords):
            elev = _interp_back(elev, len(coords))
        if elev and max(elev) == 0.0 and min(elev) == 0.0:
            raise ValueError("flat_zero")
        return elev
    except Exception:
        return [0.0 for _ in coords]

# ------------------------------
# Predefined typical routes
# ------------------------------
TYPICAL_ROUTES = {
    "Paris → Lyon": ("Paris, France", "Lyon, France"),
    "Paris → Marseille": ("Paris, France", "Marseille, France"),
    "Paris → Toulouse": ("Paris, France", "Toulouse, France"),
    "Paris → Nantes": ("Paris, France", "Nantes, France"),
    "Lyon → Marseille": ("Lyon, France", "Marseille, France"),
    "Custom": ("", "")
}

# ------------------------------
# Main UI
# ------------------------------
st.markdown("### 1) Enter your trip")

route_choice = st.selectbox("Choose a typical route", list(TYPICAL_ROUTES.keys()))

if route_choice != "Custom":
    orig_text, dest_text = TYPICAL_ROUTES[route_choice]
    st.info(f"Selected route: {route_choice}")
else:
    orig_text = ""
    dest_text = ""

col1, col2 = st.columns(2)
with col1:
    orig_text = st.text_input("Origin (address or city)", orig_text)
with col2:
    dest_text = st.text_input("Destination (address or city)", dest_text)

run_btn = st.button("Compute advised speed")

if run_btn:
    ors_key = get_ors_key()
    if not ors_key or not is_valid_ors_key(ors_key):
        st.error("Invalid OpenRouteService API key. Paste your ORS key (not an error message).")
        st.stop()

    with st.spinner("Geocoding addresses..."):
        start = ors_geocode(orig_text, ors_key)
        end = ors_geocode(dest_text, ors_key)
        if not start or not end:
            st.error("Geocoding failed. Try more precise addresses.")
            st.stop()

    # ✅ NEW: live weather override (Open-Meteo)
    if use_live_weather:
        try:
            mid_lon = (start[0] + end[0]) / 2.0
            mid_lat = (start[1] + end[1]) / 2.0
            temp_c, precip_mm_h = fetch_open_meteo_weather(mid_lat, mid_lon)
            st.info(f"🌦️ Live weather (Open-Meteo) @ route midpoint: {temp_c:.1f}°C, rain {precip_mm_h:.1f} mm/h")
        except Exception as e:
            st.warning(f"Live weather failed, using manual weather inputs. ({e})")
            # keep temp_c and precip_mm_h from sidebar

    # ✅ Recompute physics + battery factors from (possibly) updated temp_c
    rho_air = air_density_from_temp_c(temp_c)
    batt_cap_fac = battery_capacity_factor(temp_c) if use_battery_temp else 1.0
    batt_E_mult = battery_energy_multiplier(temp_c) if use_battery_temp else 1.0

    with st.spinner("Computing route..."):
        try:
            coords, length_m, duration_s = ors_route(start, end, ors_key)
            if not coords or len(coords) < 2:
                st.error("No route found.")
                st.stop()
        except Exception as e:
            st.error(f"Error while computing route: {e}")
            st.stop()

    with st.spinner("Fetching elevation profile..."):
        if not use_elevation:
            st.info("Elevation disabled - Using constant altitude")
            elevations = [0.0 for _ in coords]
        else:
            try:
                if len(coords) <= 2:
                    st.info("Short trip - Using constant elevation for simplicity")
                    elevations = [0.0 for _ in coords]
                else:
                    elevations = ors_elevation_along(coords, ors_key)
                    if len(elevations) != len(coords):
                        st.warning(f"Elevation length ({len(elevations)}) differs from coordinates ({len(coords)}). Using constant elevation.")
                        elevations = [0.0 for _ in coords]
            except Exception as e:
                st.warning(f"Elevation unavailable ({e}). Assuming constant altitude.")
                elevations = [0.0 for _ in coords]

    if len(coords) < 2:
        st.error("Invalid route: fewer than 2 points")
        st.stop()

    if len(coords) == 2:
        st.warning("⚠️ Simplified route: only 2 points (start/end)")
        st.info("Calculations will be based on a straight line. For better accuracy, try closer cities.")

        import numpy as np
        start_lon, start_lat = coords[0][0], coords[0][1]
        end_lon, end_lat = coords[1][0], coords[1][1]

        d0_m = _haversine_m(start_lon, start_lat, end_lon, end_lat)
        n_points = max(50, min(500, int(max(d0_m, 1.0) / 1000.0) * 10))

        lons = np.linspace(start_lon, end_lon, n_points)
        lats = np.linspace(start_lat, end_lat, n_points)

        coords = [[lon, lat, 0] for lon, lat in zip(lons, lats)]
        elevations = [0.0] * len(coords)

        if use_elevation:
            try:
                elev_try = ors_elevation_along(coords, ors_key)
                if isinstance(elev_try, list) and len(elev_try) == len(coords):
                    elevations = elev_try
                    st.info("Elevation profile retrieved on simplified route ✅")
            except Exception:
                pass

        length_m = d0_m
        duration_s = 0

    if len(elevations) != len(coords):
        st.error("Inconsistent data: elevation and coordinate counts differ")
        st.stop()

    st.success("Route and elevations retrieved ✅")

    # Relief
    total_up_m = 0.0
    total_down_m = 0.0
    max_abs_slope_pct = 0.0
    total_distance_m = 0.0
    slope_abs_sum_m = 0.0
    for i in range(1, len(elevations)):
        dh = elevations[i] - elevations[i-1]
        if dh > 0:
            total_up_m += dh
        elif dh < 0:
            total_down_m += -dh

        lon1, lat1 = coords[i-1][0], coords[i-1][1]
        lon2, lat2 = coords[i][0], coords[i][1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = 6371000.0 * c
        if d > 1e-3:
            slope_pct = abs((elevations[i] - elevations[i-1]) / d) * 100.0
            max_abs_slope_pct = max(max_abs_slope_pct, slope_pct)
            total_distance_m += d
            slope_abs_sum_m += abs(elevations[i] - elevations[i-1])

    mean_abs_slope_pct = (slope_abs_sum_m / total_distance_m * 100.0) if total_distance_m > 0 else 0.0

    with st.spinner("Analyzing intersections and speed limits..."):
        steps, detailed_segments = ors_route_steps(start, end, ors_key)
        intersection_data = detect_intersections_improved(steps, detailed_segments, coords)

    # HVAC
    hvac_power_kw = 0.0
    if use_climate:
        hvac_power_kw = hvac_electric_power_kw(temp_c, climate_intensity, hvac_params)
    adjusted_aux_power = aux_power_kw + hvac_power_kw

    total_mass_kg = float(mass_kg) + float(total_passenger_weight)

    # ✅ UPDATED: rain intensity → Crr factor
    rain_factor = rolling_resistance_rain_factor(precip_mm_h)
    crr_effective = float(crr) * rain_factor
    is_raining = precip_mm_h > 0.0

    battery_kwh_effective = float(battery_kwh) * float(batt_cap_fac)

    try:
        veh = dict(
            mass_kg=float(total_mass_kg),
            cda=float(cda),
            crr=float(crr_effective),
            rho_air=float(rho_air),
            eta_drive=float(eta_drive),
            regen_eff=float(regen_eff),
            aux_power_kw=float(adjusted_aux_power),
            battery_kwh=float(battery_kwh_effective),
            is_rain=bool(is_raining),
            batt_E_mult=float(batt_E_mult),
        )
        if veh["mass_kg"] <= 0 or veh["eta_drive"] <= 0 or veh["eta_drive"] > 1:
            st.error("Invalid vehicle parameters")
            st.stop()
    except (ValueError, TypeError) as e:
        st.error(f"Error in vehicle parameters: {e}")
        st.stop()

    # Limit candidate speeds by user_speed_limit
    candidates = [v for v in candidate_speeds if v <= user_speed_limit]
    if not candidates:
        candidates = [user_speed_limit]

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, v in enumerate(candidates):
        status_text.text(f"Computing for {v} km/h (with per-segment limits)...")
        progress_bar.progress((i + 1) / len(candidates))

        try:
            if use_segmented_speeds and steps and detailed_segments:
                segmented_speeds = create_segmented_speeds(coords, steps, detailed_segments, v, user_speed_limit, min_speed_delta)

                if intersection_data["slowdown_points"]:
                    for slowdown in intersection_data["slowdown_points"]:
                        step_idx = slowdown.get("step_index", 0)
                        if steps and len(steps) > 0:
                            coord_ratio = step_idx / max(len(steps), 1)
                            coord_idx = min(int(coord_ratio * (len(coords) - 1)), len(segmented_speeds) - 1)
                            if 0 <= coord_idx < len(segmented_speeds):
                                segmented_speeds[coord_idx] = max(segmented_speeds[coord_idx] * 0.7, 30)

                E_Wh, T_h, D_km = route_energy_time(coords, elevations, segmented_speeds, **veh)
                avg_speed = sum(segmented_speeds) / len(segmented_speeds) if segmented_speeds else v
            else:
                E_Wh, T_h, D_km = route_energy_time(coords, elevations, v, **veh)
                avg_speed = v

            energy_kwh_candidate = E_Wh / 1000.0
            charge_info_candidate = calculate_charging_stops(veh["battery_kwh"], energy_kwh_candidate, battery_start_pct, battery_end_pct)
            charging_time_min_candidate = charge_info_candidate["num_stops"] * CHARGING_STOP_DURATION_MIN
            total_time_min_candidate = T_h * 60.0 + charging_time_min_candidate

            results.append(dict(
                speed=v,
                energy_Wh=E_Wh,
                time_h=T_h,
                dist_km=D_km,
                avg_speed=avg_speed,
                energy_kwh=energy_kwh_candidate,
                charge_info=charge_info_candidate,
                charging_time_min=charging_time_min_candidate,
                total_time_min=total_time_min_candidate
            ))
        except Exception as e:
            st.warning(f"Error for {v} km/h: {e}")
            continue

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.error("No valid result found")
        st.stop()

    fastest_total_min = min(r["total_time_min"] for r in results)
    fastest_total_h = fastest_total_min / 60.0 if fastest_total_min else 0.0

    if fastest_total_h > 0:
        max_time_h = fastest_total_h * (1 + max_time_penalty_pct/100.0)
        feasible = [r for r in results if (r["total_time_min"] / 60.0) <= max_time_h]
        if not feasible:
            feasible = results[:]
    else:
        feasible = results[:]

    if minimize_target == "Minimize energy under time constraint":
        best = min(feasible, key=lambda r: (r["energy_Wh"], r["total_time_min"]))
    else:
        E_min, E_max = min(r["energy_Wh"] for r in feasible), max(r["energy_Wh"] for r in feasible)
        T_min, T_max = min(r["total_time_min"] for r in feasible), max(r["total_time_min"] for r in feasible)
        def norm(x, a, b): return 0.0 if a == b else (x - a) / (b - a)
        best = min(feasible, key=lambda r: norm(r["energy_Wh"], E_min, E_max) + lam * norm(r["total_time_min"], T_min, T_max))

    fastest = min(results, key=lambda r: r["total_time_min"])

    # ------------------------------
    # Output metrics
    # ------------------------------
    st.markdown("### 2) Results")

    energy_needed_kwh = best['energy_kwh']
    charge_info = best['charge_info']

    # ✅ UPDATED: cost computed using sidebar electricity price
    energy_cost = energy_needed_kwh * float(energy_cost_per_kwh)

    # ✅ NEW: CO2 estimation (kgCO2)
    grid_co2_g_per_kwh = float(grid_co2_g_per_kwh)
    trip_co2_kg = energy_needed_kwh * (grid_co2_g_per_kwh / 1000.0)  # g/kWh -> kg/kWh

    charging_time_min = best['charging_time_min']
    best_driving_time_min = best['time_h'] * 60
    best_total_time_min = best['total_time_min']

    fastest_total_time_min = fastest["total_time_min"]

    battery_start_kwh = veh["battery_kwh"] * (battery_start_pct / 100.0)
    battery_after_trip = battery_start_kwh - energy_needed_kwh
    battery_end_pct_calc = (battery_after_trip / veh["battery_kwh"]) * 100

    with st.expander("📋 Trip parameter summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Passengers", f"{num_passengers}", help=f"Total weight: {total_passenger_weight} kg")
        col2.metric("🌡️ HVAC", "✅ Yes" if use_climate else "❌ No", help=(f"Intensity: {climate_intensity}% | HVAC elec: {hvac_power_kw:.2f} kW" if use_climate else ""))
        col3.metric("🔋 Battery (start)", f"{battery_start_pct}%", help=f"{battery_start_kwh:.1f} kWh")
        col4.metric("🌦️ Weather", f"{temp_c:.1f}°C", help=f"Rain: {precip_mm_h:.1f} mm/h → Crr×{rain_factor:.2f}")

    st.info(f"📊 **Battery analysis**: Start {battery_start_pct}% ({battery_start_kwh:.1f} kWh) | After trip: {battery_end_pct_calc:.1f}% ({battery_after_trip:.1f} kWh)")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Advised speed", f"{best['speed']} km/h")
    colB.metric("Estimated energy", f"{energy_needed_kwh:.2f} kWh")
    colC.metric("Driving time", f"{best_driving_time_min:.1f} min")
    colD.metric("Total time (incl. charges)", f"{best_total_time_min:.1f} min", help=f"Ajoute {charging_time_min:.1f} min de recharge estimée")

    colR1, colR2, colR3, colR4 = st.columns(4)
    colR1.metric("Elevation gain", f"{total_up_m:.0f} m")
    colR2.metric("Elevation loss", f"{total_down_m:.0f} m")
    colR3.metric("Max slope (abs)", f"{max_abs_slope_pct:.1f} %")
    colR4.metric("Mean slope (abs)", f"{mean_abs_slope_pct:.2f} %", help="Moyenne pondérée du pourcentage de pente absolu sur l'ensemble du trajet")

    st.caption("Detected intersections/slowdowns (improved analysis)")
    col_int1, col_int2 = st.columns(2)
    col_int1.metric("Intersections", len(intersection_data["intersections"]))
    col_int2.metric("Slowdown points", len(intersection_data["slowdown_points"]))

    if use_segmented_speeds:
        st.success("✅ **Per-segment speed limits enabled**: Calculations consider limits by road type (motorway 130 km/h, city 50 km/h, etc.)")
        if 'avg_speed' in best:
            st.info(f"ℹ️ Actual average speed on the route: {best['avg_speed']:.1f} km/h (base advised speed: {best['speed']} km/h)")

    # ✅ UPDATED: add CO2 metric + keep existing cost
    colE, colF, colG, colH, colI, colJ = st.columns(6)
    colE.metric("Energy cost", f"{energy_cost:.2f} €", help=f"{energy_needed_kwh:.2f} kWh × {energy_cost_per_kwh:.2f} €/kWh")
    colF.metric("🌍 CO₂ (electricity)", f"{trip_co2_kg:.2f} kg", help=f"{energy_needed_kwh:.2f} kWh × {grid_co2_g_per_kwh:.0f} g/kWh")
    colG.metric("🔌 Required charges", f"{charge_info['num_stops']}", help=f"Nombre d'arrêts de recharge ({CHARGING_STOP_DURATION_MIN} min chacun)")
    colH.metric("Charging time", f"{charging_time_min:.1f} min")
    colI.metric("Battery after trip", f"{battery_end_pct_calc:.1f}%")
    colJ.metric("Consumption", f"{(energy_needed_kwh / best['dist_km']) if best['dist_km'] else 0:.2f} kWh/km")

    if charge_info['num_stops'] == 0:
        st.success(f"✅ **No charging needed!** You have enough battery for this trip.")
    elif charge_info['num_stops'] > 0 and charge_info['num_stops'] < 10:
        st.warning(f"🔋 **{charge_info['num_stops']} charge(s) recommended** for this trip.")
    else:
        st.error(f"⚠️ **Challenging trip**: Consumption is very high ({charge_info['num_stops']} estimated charges).")

    if battery_end_pct_calc < 20:
        st.error("⚠️ Very low battery on arrival! Consider charging before departure.")
    elif battery_end_pct_calc < 50:
        st.warning("🔋 Moderate battery level on arrival. Monitor your consumption.")

    if energy_needed_kwh > veh['battery_kwh']:
        st.error("❌ Consumption exceeds battery capacity! Trip is not feasible.")
    elif energy_needed_kwh > veh['battery_kwh'] * 0.8:
        st.warning("⚠️ High consumption. Trip is possible but risky.")

    # Map
    route_latlons = []
    for pt in coords:
        try:
            lon, lat = pt[0], pt[1]
        except (TypeError, IndexError):
            continue
        route_latlons.append((lat, lon))

    if route_latlons:
        st.markdown("#### Route map preview")
        try:
            midpoint = route_latlons[len(route_latlons) // 2]
            route_map = folium.Map(location=midpoint, zoom_start=6, tiles="CartoDB Positron", control_scale=True)
            folium.PolyLine(route_latlons, color="#1f77b4", weight=5, opacity=0.85).add_to(route_map)
            folium.Marker(route_latlons[0], tooltip="Départ", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(route_map)
            folium.Marker(route_latlons[-1], tooltip="Arrivée", icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(route_map)
            route_map.fit_bounds(route_latlons)
            st_folium(route_map, width=None, height=520, returned_objects=[])
        except Exception as map_err:
            if debug_mode:
                st.warning(f"[DEBUG] Impossible d'afficher la carte: {map_err}")

    # Savings vs fastest
    dE_Wh = best["energy_Wh"] - fastest["energy_Wh"]
    dT_min = best_total_time_min - fastest_total_time_min
    st.markdown("#### Impact vs fastest driving (among your candidate speeds)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy saved", f"{-dE_Wh/1000:.2f} kWh" if dE_Wh < 0 else f"+{dE_Wh/1000:.2f} kWh")
    c2.metric("Time added", f"{dT_min:.1f} min", help="Inclut le temps supplémentaire dû aux recharges estimées")
    # ✅ NEW: delta CO2 vs fastest
    fastest_co2_kg = float(fastest["energy_kwh"]) * (grid_co2_g_per_kwh / 1000.0)
    best_co2_kg = trip_co2_kg
    dco2 = best_co2_kg - fastest_co2_kg
    c3.metric("CO₂ vs fastest", f"{dco2:+.2f} kg")

    # Results table
    st.markdown("#### Comparison of candidate speeds")
    df = pd.DataFrame([
        dict(
            Speed_kmh=r["speed"],
            Energy_kWh=r["energy_kwh"],
            Cost_EUR=float(r["energy_kwh"]) * float(energy_cost_per_kwh),  # ✅ NEW
            CO2_kg=float(r["energy_kwh"]) * (grid_co2_g_per_kwh / 1000.0),  # ✅ NEW
            Driving_time_min=r["time_h"]*60.0,
            Charging_time_min=r["charging_time_min"],
            Total_time_min=r["total_time_min"],
            Charges=r["charge_info"]["num_stops"]
        ) for r in results
    ]).sort_values("Speed_kmh")
    st.dataframe(df, use_container_width=True)

    # Plots
    col_graph1, col_graph2 = st.columns(2)
    max_time_total_min = fastest_total_min * (1 + max_time_penalty_pct/100.0) if fastest_total_min else None
    feasible_speeds = set(
        r["speed"] for r in results
        if max_time_total_min is None or r["total_time_min"] <= max_time_total_min
    )

    with col_graph1:
        fig_energy, ax = plt.subplots(figsize=(8.5, 5.2))
        x = df["Speed_kmh"].values
        yE = df["Energy_KWh"].values if "Energy_KWh" in df.columns else df["Energy_kWh"].values  # guard
        colors = ["#2ecc71" if int(s) in feasible_speeds else "#3498db" for s in x]
        ax.plot(x, yE, color="#95a5a6", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.scatter(x, yE, s=100, c=colors, alpha=0.85, edgecolors='black', linewidth=0.8)
        ax.scatter(best["speed"], best["energy_Wh"]/1000, s=220, c="#e74c3c", marker='*', edgecolors='black', linewidth=0.8, zorder=5)
        ax.set_xlabel("Speed (km/h)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Energy (kWh)", fontsize=12, fontweight='bold')
        ax.set_title("⚡ Energy vs Speed (green = feasible under time constraint)", fontsize=13, pad=12)
        ax.grid(True, alpha=0.25, linestyle=':')
        plt.tight_layout()
        st.pyplot(fig_energy)

    with col_graph2:
        fig_time, ax = plt.subplots(figsize=(8.5, 5.2))
        yT = df["Total_time_min"].values
        ax.plot(x, yT, color="#95a5a6", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.scatter(x, yT, s=100, c=["#2ecc71" if int(s) in feasible_speeds else "#e67e22" for s in x], alpha=0.85, edgecolors='black', linewidth=0.8)
        ax.scatter(best["speed"], best_total_time_min, s=220, c="#e74c3c", marker='*', edgecolors='black', linewidth=0.8, zorder=5)
        if max_time_total_min is not None:
            ax.axhline(max_time_total_min, color="#2ecc71", linestyle=":", linewidth=1.2, alpha=0.8, label="Max allowed total time")
            ax.legend(loc="best", fontsize=10)
        ax.set_xlabel("Speed (km/h)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Total time (minutes)", fontsize=12, fontweight='bold')
        ax.set_title("⏱️ Total time vs Speed (green = feasible)", fontsize=13, pad=12)
        ax.grid(True, alpha=0.25, linestyle=':')
        plt.tight_layout()
        st.pyplot(fig_time)

else:
    st.info("Enter an origin and a destination, provide your ORS key, then click *Compute advised speed*.")
