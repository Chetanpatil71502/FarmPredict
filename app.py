import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmPredict – AI Crop Advisor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_v2.pkl")
FEATURES   = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Crop metadata ──────────────────────────────────────────────────────────────
CROP_META = {
    # Cereals & Staples
    "rice":         ("🍚", "Kharif",           "Staple grain; thrives in warm, waterlogged conditions. Leading crop in eastern & southern India."),
    "wheat":        ("🌾", "Rabi",             "Second-largest staple crop. Prefers cool weather and well-drained loamy soil."),
    "maize":        ("🌽", "Kharif / Rabi",    "Versatile cereal used for food, feed & starch. Grows across diverse agro-climatic zones."),
    "sugarcane":    ("🎋", "Annual",           "Major cash crop and source of sugar & ethanol. High water and nutrient demand."),
    "cotton":       ("🌿", "Kharif",           "Key fibre cash crop; prefers deep black cotton soil. Grown extensively in Maharashtra & Gujarat."),
    "jute":         ("🌿", "Kharif",           "Golden fibre crop requiring warm, humid conditions. Predominantly grown in West Bengal & Assam."),
    # Millets
    "bajra":        ("🌾", "Kharif",           "Pearl millet — highly drought-tolerant; suited for arid Rajasthan & Gujarat regions."),
    "jowar":        ("🌾", "Kharif / Rabi",    "Sorghum — staple coarse grain of the Deccan plateau; excellent drought resistance."),
    "ragi":         ("🌾", "Kharif",           "Finger millet — rich in calcium; major crop of Karnataka, Andhra Pradesh & Tamil Nadu."),
    "barley":       ("🌾", "Rabi",             "Cool-season cereal used for malt, animal feed & traditional cuisine."),
    "amaranth":     ("🌿", "Kharif",           "Nutritious pseudocereal; grows in semi-arid areas with minimal inputs."),
    # Pulses
    "chickpea":     ("🫘", "Rabi",             "Largest pulse crop of India; high protein, drought-tolerant. Grown in MP, Rajasthan & Maharashtra."),
    "kidneybeans":  ("🫘", "Kharif",           "Protein-rich rajma; popular in North India. Improves soil nitrogen naturally."),
    "pigeonpeas":   ("🫘", "Kharif",           "Arhar/tur dal — second-most important pulse. Drought-tolerant perennial legume."),
    "mothbeans":    ("🫘", "Kharif",           "Highly heat- and drought-tolerant pulse; staple in Rajasthan's arid zones."),
    "mungbean":     ("🫘", "Kharif",           "Green gram — short-duration pulse; easy to grow with low water requirement."),
    "blackgram":    ("🫘", "Kharif",           "Urad dal — widely used in South Indian cuisine; suitable for low-fertility soils."),
    "lentil":       ("🌾", "Rabi",             "Masoor dal — cool-season pulse; excellent source of protein and iron."),
    "soybean":      ("🫘", "Kharif",           "Oilseed-cum-pulse; major crop in Madhya Pradesh. Used for oil, meal & tofu."),
    "peas":         ("🫛", "Rabi",             "Cool-season vegetable-cum-pulse. High demand in fresh, frozen and dried forms."),
    "groundnut":    ("🥜", "Kharif",           "Peanut — India's largest oilseed crop. Grows well in light sandy loam soils."),
    # Oilseeds
    "sunflower":    ("🌻", "Kharif / Rabi",    "High-yield oilseed; adaptable to diverse climates. Grown in Karnataka & Andhra Pradesh."),
    "mustard":      ("🌼", "Rabi",             "Rapeseed-mustard — second-largest oilseed; cultivated in the Indo-Gangetic plains during winter."),
    "sesame":       ("🌿", "Kharif",           "Til — one of the oldest oilseeds; high heat tolerance. Grown in Gujarat, Rajasthan & MP."),
    # Fruits
    "banana":       ("🍌", "Perennial",        "India is the world's largest banana producer. High water and nutrient demand."),
    "mango":        ("🥭", "Perennial",        "King of fruits; India accounts for ~40% of global production. Prefers hot, dry climate."),
    "apple":        ("🍎", "Perennial",        "Cool-climate fruit grown in J&K, Himachal Pradesh & Uttarakhand at high elevations."),
    "grapes":       ("🍇", "Perennial",        "India is a leading grape exporter. Nashik (Maharashtra) is the wine & grape capital."),
    "watermelon":   ("🍉", "Zaid",             "High water-content summer fruit. Grows rapidly in warm, well-drained sandy loam."),
    "muskmelon":    ("🍈", "Zaid",             "Kharbuja — warm-season fruit; favours dry climate during ripening."),
    "orange":       ("🍊", "Perennial",        "Nagpur orange is a GI-tagged variety. Requires warm days and cool nights."),
    "papaya":       ("🍑", "Perennial",        "Fast-growing tropical fruit with year-round yield. India is the top global producer."),
    "coconut":      ("🥥", "Perennial",        "Kalpavriksha — thrives in humid coastal tropics. Major crop of Kerala, Karnataka & Tamil Nadu."),
    "pomegranate":  ("🍎", "Perennial",        "Hardy fruit suited to dry and semi-arid climates. India is the world's largest producer."),
    # Vegetables
    "tomato":       ("🍅", "Kharif / Rabi",    "Most widely grown vegetable; India is the second-largest producer globally."),
    "potato":       ("🥔", "Rabi",             "Third most important food crop. Uttar Pradesh & West Bengal are leading producers."),
    "onion":        ("🧅", "Rabi / Kharif",    "India is the largest onion exporter. Maharashtra's Nashik is the onion capital."),
    "garlic":       ("🧄", "Rabi",             "Widely grown in Madhya Pradesh & Rajasthan. High medicinal and culinary value."),
    "chilli":       ("🌶️", "Kharif",           "India is the world's largest producer & exporter of chillies. Andhra Pradesh leads production."),
    "brinjal":      ("🍆", "Kharif",           "Baingan — one of India's oldest cultivated vegetables. Grows across all agro-climatic zones."),
    "okra":         ("🌿", "Kharif",           "Bhindi — heat-loving vegetable popular across India. Quick-maturing and high-yielding."),
    "cauliflower":  ("🥦", "Rabi",             "Cool-season brassica; India is the second-largest global producer."),
    "cabbage":      ("🥬", "Rabi",             "Cool-season vegetable grown extensively in hills and northern plains during winter."),
    "bottle_gourd": ("🥒", "Kharif",           "Lauki — low-calorie summer vegetable; widely grown across Indo-Gangetic plains."),
    "bitter_gourd": ("🥒", "Kharif",           "Karela — popular medicinal vegetable; grows in hot, humid conditions."),
    # Spices
    "turmeric":     ("🟡", "Kharif",           "Haldi — India produces 80% of global supply. Andhra Pradesh, Tamil Nadu & Odisha lead."),
    "ginger":       ("🫚", "Kharif",           "India is the second-largest ginger producer. High humidity and well-drained soil needed."),
    "cardamom":     ("🌿", "Perennial",        "Queen of spices — grown in the cardamom hills of Kerala & Karnataka at high altitude."),
    "pepper":       ("🌿", "Perennial",        "King of spices — India's oldest exported spice. Thrives in humid tropical forests of Kerala."),
    "coffee":       ("☕", "Perennial",        "Shade-grown in Karnataka's Coorg & Chikmagalur. India is the 6th largest coffee producer."),
}

CROP_CATEGORIES = {
    "🌾 Cereals & Millets": ["rice","wheat","maize","bajra","jowar","ragi","barley","amaranth"],
    "🫘 Pulses": ["chickpea","kidneybeans","pigeonpeas","mothbeans","mungbean","blackgram","lentil","soybean","peas","groundnut"],
    "💰 Cash Crops": ["sugarcane","cotton","jute"],
    "🫒 Oilseeds": ["sunflower","mustard","sesame"],
    "🍎 Fruits": ["banana","mango","apple","grapes","watermelon","muskmelon","orange","papaya","coconut","pomegranate"],
    "🥦 Vegetables": ["tomato","potato","onion","garlic","chilli","brinjal","okra","cauliflower","cabbage","bottle_gourd","bitter_gourd"],
    "🌶️ Spices & Plantation": ["turmeric","ginger","cardamom","pepper","coffee"],
}

# ── Styles ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---- Global font & body ---- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Hide Streamlit chrome ---- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Hero banner ---- */
    .hero {
        background: linear-gradient(135deg, #1a3c1a 0%, #2d6a2d 60%, #4a8f3f 100%);
        border-radius: 20px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "🌾";
        position: absolute;
        right: 32px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 88px;
        opacity: 0.18;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #f0f9e8;
        margin: 0 0 6px;
        line-height: 1.1;
    }
    .hero-sub {
        color: #a8d5a2;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 0;
        max-width: 500px;
    }
    .badge-row {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #d4edca;
        font-weight: 500;
    }

    /* ---- Section labels ---- */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b9e6b;
        margin: 24px 0 10px;
    }

    /* ---- Result card ---- */
    .result-card {
        background: linear-gradient(150deg, #f0faf0 0%, #e8f5e9 100%);
        border: 2px solid #4caf50;
        border-radius: 18px;
        padding: 32px 36px;
        text-align: center;
        margin: 8px 0 16px;
        box-shadow: 0 4px 24px rgba(76,175,80,0.12);
    }
    .result-emoji  { font-size: 80px; line-height: 1.1; margin-bottom: 4px; }
    .result-name   { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #1b5e20; margin: 0; }
    .result-season { font-size: 0.9rem; color: #388e3c; margin: 8px 0 4px; font-weight: 500; }
    .result-desc   { font-size: 0.88rem; color: #4a6b4a; max-width: 480px; margin: 0 auto; line-height: 1.55; }

    /* ---- Confidence bar label ---- */
    .conf-label {
        font-size: 0.82rem;
        color: #555;
        margin: 4px 0 8px;
        font-weight: 500;
    }

    /* ---- Alt crops ---- */
    .alt-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 24px;
        padding: 6px 14px;
        font-size: 0.84rem;
        margin: 4px 4px 4px 0;
        color: #333;
    }
    .alt-pct { color: #888; font-size: 0.78rem; }

    /* ---- Advisory ---- */
    .advisory {
        background: #fffde7;
        border-left: 4px solid #f9a825;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #5d4e20;
        margin-top: 16px;
        line-height: 1.5;
    }

    /* ---- Input card ---- */
    .input-section {
        background: #fafafa;
        border: 1px solid #ebebeb;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    /* ---- Predict button ---- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2d6a2d, #4caf50) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 14px 20px !important;
        letter-spacing: 0.02em !important;
        transition: transform 0.1s, box-shadow 0.1s !important;
        box-shadow: 0 4px 14px rgba(45,106,45,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(45,106,45,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">FarmPredict</p>
    <p class="hero-sub">Enter your soil test values and local climate data to get an AI-powered crop recommendation.</p>
    <div class="badge-row">
        <span class="badge">50 Indian Crops</span>
        <span class="badge">Random Forest · 90%+ Accuracy</span>
        <span class="badge">7 Soil & Climate Inputs</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input Form ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Soil Nutrients (kg/ha)</div>', unsafe_allow_html=True)
st.caption("Values from your soil test report — typically available from KVK or soil health card.")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=60,
                            help="Nitrogen kg/ha — range 0–140")
    with col2:
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50,
                            help="Phosphorus kg/ha — range 5–145")
    with col3:
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=40,
                            help="Potassium kg/ha — range 5–205")

st.markdown('<div class="section-label">Climate & Soil Chemistry</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=24.0, step=0.5,
                            help="Mean seasonal temperature in °C")
    humidity = st.slider("Humidity (%)", min_value=10.0, max_value=99.0, value=70.0, step=0.5,
                         help="Relative humidity in %")
with col5:
    ph = st.slider("Soil pH", min_value=3.0, max_value=9.9, value=6.5, step=0.1,
                   help="Soil pH — 6.0–7.5 is ideal for most crops")
    rainfall = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0, step=5.0,
                         help="Expected seasonal/annual rainfall in mm")

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("")
if st.button("🌱  Analyse & Recommend Crop", type="primary", use_container_width=True):
    input_df   = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
    crop       = model.predict(input_df)[0]
    proba_arr  = model.predict_proba(input_df)[0]
    confidence = round(float(np.max(proba_arr)) * 100, 1)

    emoji, season, desc = CROP_META.get(crop, ("🌱", "—", ""))

    st.markdown(f"""
    <div class="result-card">
        <div class="result-emoji">{emoji}</div>
        <p class="result-name">{crop.replace("_", " ").title()}</p>
        <p class="result-season">📅 {season}</p>
        <p class="result-desc">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="conf-label">Model confidence — {confidence}%</div>', unsafe_allow_html=True)
    st.progress(int(confidence))

    # Top-3 alternatives
    top3_idx = np.argsort(proba_arr)[::-1][1:4]
    alts = [(model.classes_[i], round(proba_arr[i]*100,1)) for i in top3_idx if proba_arr[i] > 0.01]
    if alts:
        st.markdown('<div class="section-label">Other Possible Crops</div>', unsafe_allow_html=True)
        chips = ""
        for alt_crop, alt_conf in alts:
            alt_emoji = CROP_META.get(alt_crop, ("🌱",))[0]
            chips += f'<span class="alt-chip">{alt_emoji} {alt_crop.replace("_"," ").title()} <span class="alt-pct">{alt_conf}%</span></span>'
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("""
    <div class="advisory">
        ⚠️ <strong>Advisory:</strong> This recommendation is generated by a machine learning model trained on
        simulated agronomic data for Indian conditions. Always cross-check with your local
        Krishi Vigyan Kendra (KVK), ICAR extension officer, or certified agronomist before sowing.
    </div>
    """, unsafe_allow_html=True)

# ── Crop Reference ─────────────────────────────────────────────────────────────
st.markdown("")
st.markdown('<div class="section-label">Supported Crops — Quick Reference</div>', unsafe_allow_html=True)
for category, crops_list in CROP_CATEGORIES.items():
    with st.expander(f"{category}  ({len(crops_list)} crops)"):
        for c in crops_list:
            emoji, season, desc = CROP_META.get(c, ("🌱","—",""))
            st.markdown(f"**{emoji} {c.replace('_',' ').title()}** · *{season}*  \n{desc}")
            st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin:32px 0 16px; border-color:#e5e5e5;">
<p style="text-align:center; font-size:0.8rem; color:#aaa;">
    FarmPredict · Powered by Random Forest · 50 Indian Crops · For educational & advisory use only
</p>
""", unsafe_allow_html=True)
