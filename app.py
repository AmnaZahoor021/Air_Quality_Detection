# ============================================================
# AI AIR QUALITY DETECTION — Neon Dashboard UI
# Updated: Air sign added to the left of "Detection", 0.00 defaults, custom backgrounds.
# ============================================================

import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AI Air Quality Detection",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_air_quality_classifier.joblib"
SIDEBAR_IMAGE_PATH = BASE_DIR / "sidebar_bg.png"
MAIN_BG_IMAGE_PATH = BASE_DIR / "mainpage_bg.png"


def render_html(content: str) -> None:
    """Render HTML directly with Streamlit."""
    st.markdown(content, unsafe_allow_html=True)


def get_image_base64(path: Path):
    """Return a local image as Base64, if it exists."""
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    return None


@st.cache_resource
def load_model():
    """Load the existing trained Random Forest model."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


model = load_model()
sidebar_image = get_image_base64(SIDEBAR_IMAGE_PATH)
main_bg_image = get_image_base64(MAIN_BG_IMAGE_PATH)

# Base64 background styling for the main app using mainpage_bg.png
main_bg_css = ""
if main_bg_image:
    main_bg_css = f"""
    .stApp {{
        background-image: linear-gradient(rgba(2, 11, 18, 0.75), rgba(2, 11, 18, 0.85)),
        url("data:image/png;base64,{main_bg_image}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }}
    """
else:
    main_bg_css = """
    .stApp {
        background: 
          radial-gradient(ellipse at 85% 10%, rgba(0, 150, 200, 0.18), transparent 45%),
          radial-gradient(ellipse at 15% 85%, rgba(0, 80, 130, 0.15), transparent 45%),
          linear-gradient(135deg, #020b12 0%, #03131e 50%, #01070c 100%);
    }
    """

# -------------------- Exact Neon Theme CSS --------------------
render_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

{main_bg_css}

header[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{ max-width: 1550px; padding: 1.5rem 2rem 2rem; }}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(2,15,24,0.92) 0%, rgba(1,9,15,0.97) 100%);
    border-right: 1px solid rgba(0, 200, 240, 0.25);
    width: 290px !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1.5rem; }}

/* Force Radio Button Indicator to Neon Blue */
div[data-testid="stRadio"] input[type="radio"] {{
    accent-color: #00c8f0 !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] input {{
    accent-color: #00c8f0 !important;
}}
div[data-testid="stRadio"] > label {{ display: none; }}
div[data-testid="stRadio"] > div {{ gap: 6px; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    background: transparent;
    border: 1px solid transparent;
    padding: 12px 18px;
    border-radius: 10px;
    color: #a3c7d7;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
    transition: all 0.25s ease;
    width: 100%;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
    background: rgba(0, 123, 167, 0.25);
    color: #ffffff;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
    background: linear-gradient(90deg, #007ba7 0%, #004f82 100%);
    border: 1px solid rgba(0,200,240,0.45);
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(0,200,240,0.2);
}}

/* Hero Section */
.hero-container {{
    position: relative;
    padding: 10px 0 20px 0;
    margin-bottom: 10px;
}}
.eyebrow {{
    color: #00c8f0;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.hero-title {{
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 10px 0;
    line-height: 1.1;
}}
.hero-title span {{ color: #00c8f0; }}
.hero-desc {{
    color: #b8d9e8;
    font-size: 15px;
    margin: 0;
    max-width: 720px;
    line-height: 1.6;
}}
.top-right-art {{
    position: absolute;
    right: 10px;
    top: 0px;
    text-align: right;
}}
.slogan-top {{
    color: #00c8f0;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
    line-height: 1.4;
    margin-bottom: 10px;
}}

/* Glass Panels */
.neon-panel {{
    background: linear-gradient(145deg, rgba(3, 25, 37, 0.88), rgba(1, 14, 22, 0.94));
    border: 1px solid rgba(0, 174, 220, 0.55);
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 0 22px rgba(0, 116, 163, 0.12), inset 0 0 20px rgba(0, 108, 150, 0.04);
    height: 100%;
    backdrop-filter: blur(6px);
}}
.panel-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    padding-bottom: 12px;
    margin-bottom: 18px;
    border-bottom: 1px solid rgba(0, 200, 240, 0.45);
}}

/* Number Inputs & Blue Stepper Buttons Hover */
div[data-testid="stNumberInput"] label {{
    color: #bdeaff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    margin-bottom: 6px !important;
    white-space: nowrap !important;
}}
div[data-testid="stNumberInput"] input {{
    background-color: rgba(4, 28, 42, 0.85) !important;
    color: #ffffff !important;
    border: 1px solid #007ba7 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
}}
div[data-testid="stNumberInput"] input:focus {{
    border-color: #00c8f0 !important;
    box-shadow: 0 0 0 1px #00c8f0 !important;
}}
div[data-testid="stNumberInput"] button {{
    background: rgba(4, 28, 42, 0.8) !important;
    color: #bdeaff !important;
    border-color: #007ba7 !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stNumberInput"] button:hover {{
    background: rgba(0, 200, 240, 0.3) !important;
    color: #00c8f0 !important;
    border-color: #00c8f0 !important;
    box-shadow: 0 0 10px rgba(0, 200, 240, 0.4) !important;
}}
div[data-testid="stNumberInput"] button:active {{
    background: #00c8f0 !important;
    color: #000000 !important;
}}

/* Predict Button Style (Dark Blue Background & White Text) */
div.stButton > button {{
    width: 100%;
    min-height: 54px;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    border: 1px solid rgba(0, 200, 240, 0.4) !important;
    border-radius: 10px !important;
    background: linear-gradient(100deg, #007ba7 0%, #004f82 100%) !important;
    box-shadow: 0 4px 18px rgba(0, 116, 163, 0.3) !important;
    transition: all 0.25s ease !important;
    margin-top: 15px !important;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(0, 200, 240, 0.4) !important;
    background: linear-gradient(100deg, #0093c4 0%, #00609a 100%) !important;
    border: 1px solid #00c8f0 !important;
}}

/* AQI Circular Gauge */
.aqi-gauge {{
    width: 190px;
    height: 190px;
    margin: 15px auto;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #ff168d 0deg, #b92ed3 95deg, #007ca8 205deg, #00b6db 270deg, #ff168d 360deg);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 22px rgba(0, 166, 210, 0.18);
}}
.aqi-inner {{
    width: 164px;
    height: 164px;
    border-radius: 50%;
    background: #04131e;
    border: 2px solid #0c3445;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}
.aqi-label {{
    color: #ff2d95;
    font-size: 15px;
    font-weight: 800;
}}
.aqi-value {{
    color: #ff2d95;
    font-size: 43px;
    font-weight: 900;
    line-height: 1.25;
}}
.category {{
    text-align: center;
    font-size: 15px;
    font-weight: 800;
    border: 1px solid;
    border-radius: 30px;
    padding: 10px 12px;
    margin: 16px 0;
}}
.advice {{
    background: rgba(3, 29, 42, 0.85);
    border-left: 4px solid #00c8f0;
    border-radius: 8px;
    padding: 13px;
    line-height: 1.65;
    color: #d4e8f1;
    font-size: 13px;
    text-align: left;
}}
.scale {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 6px;
    margin-top: 12px;
}}
.scale-color {{ height: 11px; border-radius: 5px; }}
.scale-label {{
    font-size: 9.5px;
    text-align: center;
    color: #c6e2ed;
    padding-top: 6px;
    line-height: 1.35;
    overflow-wrap: anywhere;
}}
.footer {{
    border-top: 1px solid #16475a;
    padding-top: 18px;
    margin-top: 25px;
    text-align: center;
    color: #6fcce5;
    font-size: 12px;
    letter-spacing: .4px;
}}
</style>
""")

# Apply local sidebar background image if present[cite: 1]
if sidebar_image:
    render_html(f"""
    <style>
    section[data-testid="stSidebar"] > div {{
        background-image: linear-gradient(rgba(2,15,24,.67), rgba(2,10,17,.83)),
        url("data:image/png;base64,{sidebar_image}");
        background-size: cover;
        background-position: center bottom;
    }}
    </style>
    """)

# -------------------- Sidebar Component --------------------
with st.sidebar:
    render_html("""
    <div style="padding: 4px 0 16px;">
      <div style="font-size: 38px; line-height: 1; color: #ff2d95; text-shadow: 0 0 18px rgba(255,45,149,0.4);">🍃</div>
      <div style="font-size: 24px; font-weight: 900; line-height: 1.18; margin-top: 10px; color: #f4f8fc;">
        AI Air Quality<br><span style="color: #00c8f0;">Detection</span>
      </div>
      <div style="font-size: 12px; margin-top: 8px; color: #bdeaff;">Cleaner Air • Healthier Tomorrow</div>
    </div>
    <div style="height: 1px; background: linear-gradient(90deg, #00c8f0, transparent); margin-bottom: 20px;"></div>
    """)

    page = st.radio(
        "Navigation",
        options=["🏠 Home", "ℹ️ About", "⚙️ How It Works"],
        index=0,
    )

    render_html("""
    <div style="margin-top: 160px; text-align: center;">
      <div style="color: #00c8f0; font-size: 11px; font-weight: 900; letter-spacing: 2px; line-height: 1.6;">
        BETTER DATA<br>CLEANER AIR
      </div>
    </div>
    """)


# -------------------- AQI categories and data helpers --------------------
CITY_NAMES = [
    'Faisalabad', 'Islamabad', 'Karachi', 'Lahore', 'Larkana',
    'Multan', 'Peshawar', 'Quetta', 'Rawalpindi'
]
POLLUTANT_COLUMNS = ['CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3']
CITY_DATA_DIR = BASE_DIR / 'dataset'

AQI_CATEGORIES = {
    1: ('Good', '#00d084', 'Air quality is good. Outdoor activities can generally continue as usual.'),
    2: ('Fair', '#a8d84e', 'Air quality is generally acceptable. People who are unusually sensitive may prefer to reduce long outdoor exertion if they feel discomfort.'),
    3: ('Moderate', '#ffd84d', 'Some people who are sensitive to air pollution may experience discomfort. Consider taking breaks from prolonged outdoor activity.'),
    4: ('Poor', '#ff9b42', 'Air pollution may affect health, especially for children, older adults, and people with breathing or heart conditions. Reduce prolonged outdoor exertion.'),
    5: ('Very Poor', '#ff2d55', 'Air pollution is at a very high level. Limit outdoor activity, keep windows closed when practical, and follow local health guidance.'),
}

@st.cache_data(show_spinner=False)
def load_city_dataset(city_name):
    csv_path = CITY_DATA_DIR / f'historical_air_pollution_all_{city_name.lower()}.csv'

    if not csv_path.exists():
        raise FileNotFoundError(
            f'Dataset not found for {city_name}: {csv_path.name}'
        )

    data = pd.read_csv(csv_path)

    data['Timestamp'] = pd.to_datetime(
        data['Timestamp'], errors='coerce'
    )

    for column in POLLUTANT_COLUMNS + ['AQI']:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Treat invalid measurements as missing values.
    data = data.replace(-9999, float('nan'))

    # Remove rows without a valid timestamp.
    # The trained model pipeline will handle missing pollutants.
    data = data.dropna(subset=['Timestamp'])

    return data.sort_values('Timestamp')


def predict_aqi(city_name, pollutant_values):
    if model is None:
        raise RuntimeError(f'The trained model could not be loaded from {MODEL_PATH}. Check that the model file exists and that your scikit-learn version matches the one used to train it.')
    row = {'City': city_name}
    row.update(pollutant_values)
    prediction = int(round(float(model.predict(pd.DataFrame([row]))[0])))
    return max(1, min(5, prediction))


def render_result(prediction, city_name, source_text):
    category, category_color, advice = AQI_CATEGORIES[prediction]
    render_html(f"""
    <div class="neon-panel" style="text-align:center;">
      <div style="color:#8edcf0; letter-spacing:1.8px; font-size:11px; font-weight:800;">RANDOM FOREST PREDICTION</div>
      <div class="aqi-gauge"><div class="aqi-inner">
        <div class="aqi-label">AQI LEVEL</div>
        <div class="aqi-value">{prediction}/5</div>
      </div></div>
      <div class="category" style="color:{category_color}; border-color:{category_color};">{category}</div>
      <div style="color:#b8d9e8; font-size:13px; margin:5px 0 12px;">City: <b>{city_name}</b></div>
      <div class="advice"><b>ⓘ &nbsp; What this means</b><br>{advice}</div>
      <div style="color:#8db8c9; font-size:11px; margin-top:13px;">{source_text}</div>
    </div>
    """)


def show_aqi_scale():
    render_html("""
    <div class="neon-panel" style="padding:18px; margin-top:22px;">
      <div style="font-size:15px; font-weight:800; color:#fff;">Air Quality Scale (1–5)</div>
      <div class="scale">
        <div class="scale-color" style="background:#00d084"></div>
        <div class="scale-color" style="background:#a8d84e"></div>
        <div class="scale-color" style="background:#ffd84d"></div>
        <div class="scale-color" style="background:#ff9b42"></div>
        <div class="scale-color" style="background:#ff2d55"></div>
      </div>
      <div class="scale">
        <div class="scale-label">1<br>Good</div>
        <div class="scale-label">2<br>Fair</div>
        <div class="scale-label">3<br>Moderate</div>
        <div class="scale-label">4<br>Poor</div>
        <div class="scale-label">5<br>Very Poor</div>
      </div>
    </div>
    """)


# -------------------- HOME PAGE --------------------
if "Home" in page:
    render_html("""
    <div class="hero-container">
      <div class="top-right-art">
        <div class="slogan-top">CLEANER AIR<br>HEALTHIER LIVES</div>
        <svg width="70" height="35" viewBox="0 0 70 35" fill="none" style="opacity:0.9;">
          <path d="M5 12 Q22 2,40 12 T68 12" stroke="#00c8f0" stroke-width="2.5" fill="none"/>
          <path d="M10 24 Q27 14,45 24 T65 24" stroke="#ff2d95" stroke-width="2.5" fill="none"/>
        </svg>
      </div>
      <div class="eyebrow">AI POWERED AIR QUALITY MONITORING</div>
      <h1 class="hero-title">AI Air Quality <span>Detection 🍃</span></h1>
      <p class="hero-desc">Check air quality in nine Pakistani cities. Enter pollutant measurements yourself or choose a city for a simple automatic estimate.</p>
    </div>
    """)

    mode = st.radio(
        'Choose prediction mode',
        ['Simple City Check', 'Manual Pollutant Input'],
        horizontal=True,
        help='Simple City Check uses the latest available record in the selected city dataset. Manual Pollutant Input lets you enter the measurements yourself.'
    )
    input_col, result_col = st.columns([1.75, 1], gap='large')
    prediction = None
    selected_city = None
    source_text = ''

    with input_col:
        if mode == 'Simple City Check':
            render_html("""
            <div class="neon-panel" style="margin-bottom:12px;">
              <div class="panel-header">🌍 Simple City Check</div>
              <div style="font-size:13px; color:#b8d9e8; line-height:1.6;">Select your city and press the button. The system will load the latest available pollutant measurements from that city's dataset and run the AI model automatically.</div>
            </div>
            """)
            selected_city = st.selectbox('Select your city', CITY_NAMES, key='simple_city')
            run_simple = st.button('✦  CHECK MY CITY AIR QUALITY', use_container_width=True)
            if run_simple:
                try:
                    city_data = load_city_dataset(selected_city)
                    if city_data.empty:
                        st.error('No valid measurements were found for this city.')
                    else:
                        latest = city_data.iloc[-1]
                        values = {column: float(latest[column]) for column in POLLUTANT_COLUMNS}
                        prediction = predict_aqi(selected_city, values)
                        record_time = latest['Timestamp'].strftime('%d %B %Y, %I:%M %p')
                        source_text = f'Based on the latest dataset record: {record_time}. This is historical data, not live monitoring.'
                        with st.expander('View pollutant values used for this prediction'):
                            st.dataframe(pd.DataFrame([values]), use_container_width=True, hide_index=True)
                        st.info(f'Dataset record used: {record_time}')
                except Exception as error:
                    st.error(f'Could not generate the city prediction: {error}')
        else:
            render_html("""
            <div class="neon-panel" style="margin-bottom:12px;">
              <div class="panel-header">🧪 Manual Pollutant Input</div>
              <div style="font-size:13px; color:#b8d9e8; line-height:1.6;">Choose your city and enter the eight pollutant measurements. The AI model will estimate the AQI category from your values.</div>
            </div>
            """)
            selected_city = st.selectbox('Select your city', CITY_NAMES, key='manual_city')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                co = st.number_input('CO', min_value=0.0, value=0.0, step=1.0, format='%.2f')
                no = st.number_input('NO', min_value=0.0, value=0.0, step=1.0, format='%.2f')
                no2 = st.number_input('NO₂', min_value=0.0, value=0.0, step=1.0, format='%.2f')
                o3 = st.number_input('O₃', min_value=0.0, value=0.0, step=1.0, format='%.2f')
            with c2:
                so2 = st.number_input('SO₂', min_value=0.0, value=0.0, step=1.0, format='%.2f')
                pm25 = st.number_input('PM2.5', min_value=0.0, value=0.0, step=1.0, format='%.2f')
                pm10 = st.number_input('PM10', min_value=0.0, value=0.0, step=1.0, format='%.2f')
                nh3 = st.number_input('NH₃', min_value=0.0, value=0.0, step=1.0, format='%.2f')
            run_manual = st.button('✦  PREDICT AIR QUALITY', use_container_width=True)
            if run_manual:
                try:
                    values = {'CO': co, 'NO': no, 'NO2': no2, 'O3': o3, 'SO2': so2, 'PM2.5': pm25, 'PM10': pm10, 'NH3': nh3}
                    prediction = predict_aqi(selected_city, values)
                    source_text = 'Based on the pollutant values entered manually.'
                except Exception as error:
                    st.error(f'Prediction failed: {error}')
        show_aqi_scale()

    with result_col:
        render_html("""
        <div class="panel-header" style="border-bottom:none; margin-bottom:0; padding-bottom:4px;">
          <span style="color:#00c8f0; font-size:25px;">🍃</span> Air Quality Result
        </div>
        <div style="font-size:11px; color:#86d8ed; font-weight:600; margin-left:34px; letter-spacing:.6px; margin-bottom:12px;">POWERED BY RANDOM FOREST</div>
        """)
        if prediction is None:
            render_html("""
            <div class="neon-panel" style="text-align:center; padding:35px 20px;">
              <div style="font-size:48px;">🌫️</div>
              <div style="color:#fff; font-size:19px; font-weight:800; margin:12px 0;">Your result will appear here</div>
              <div style="color:#b8d9e8; font-size:13px; line-height:1.6;">Choose a prediction mode, provide the required information, and press the prediction button.</div>
            </div>
            """)
        else:
            render_result(prediction, selected_city, source_text)
        st.caption('This is an AI estimate based on the supplied historical dataset. It is not an official monitoring-station reading or a live air-quality alert.')

# -------------------- ABOUT PAGE --------------------
elif 'About' in page:
    render_html("""
    <div class="hero-container">
      <div class="eyebrow">PROJECT OVERVIEW & MISSION</div>
      <h1 class="hero-title">About <span>AI Air Quality Detection</span></h1>
      <p class="hero-desc">A machine-learning project that helps users understand air quality across nine Pakistani cities.</p>
    </div>
    <div class="neon-panel" style="margin-bottom:15px;">
      <div class="panel-header">🌍 Project Overview</div>
      <p style="font-size:15px; line-height:1.7; color:#d4e8f1;">The application uses historical pollutant measurements and a Random Forest classifier to estimate an air-quality category from 1 to 5. It offers a simple city-based mode as well as a manual mode for users who have pollutant measurements.</p>
    </div>
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:15px;">
      <div class="neon-panel"><div class="panel-header" style="font-size:17px;">🧠 Random Forest Classifier</div><p style="font-size:14px; color:#b8d9e8; line-height:1.6;">Uses pollutant measurements and the selected city to classify air quality into five levels.</p></div>
      <div class="neon-panel"><div class="panel-header" style="font-size:17px;">🏙️ Nine Cities</div><p style="font-size:14px; color:#b8d9e8; line-height:1.6;">Supports Faisalabad, Islamabad, Karachi, Lahore, Larkana, Multan, Peshawar, Quetta, and Rawalpindi.</p></div>
      <div class="neon-panel"><div class="panel-header" style="font-size:17px;">👥 Two Prediction Modes</div><p style="font-size:14px; color:#b8d9e8; line-height:1.6;">Simple City Check for users who only want to select a city, and Manual Pollutant Input for users who know their measurements.</p></div>
    </div>
    """)

# -------------------- HOW IT WORKS PAGE --------------------
elif 'How It Works' in page:
    render_html("""
    <div class="hero-container">
      <div class="eyebrow">UNDER THE HOOD</div>
      <h1 class="hero-title">How <span>It Works</span></h1>
      <p class="hero-desc">Two ways to estimate air quality using the same trained Random Forest model.</p>
    </div>
    <div class="neon-panel" style="margin-bottom:15px;">
      <div class="panel-header">1️⃣ Simple City Check</div>
      <p style="font-size:14px; line-height:1.6; color:#d4e8f1;">The user selects a city. The application loads the latest valid pollutant measurements available in that city's CSV file, sends the values and city name to the trained model, and displays the predicted category. The timestamp is shown so users know how old the source record is.</p>
    </div>
    <div class="neon-panel" style="margin-bottom:15px;">
      <div class="panel-header">2️⃣ Manual Pollutant Input</div>
      <p style="font-size:14px; line-height:1.6; color:#d4e8f1;">The user selects a city and enters CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, and NH₃ values. The same model predicts the 1–5 air-quality category.</p>
    </div>
    <div class="neon-panel">
      <div class="panel-header">3️⃣ AQI Category & Guidance</div>
      <p style="font-size:14px; line-height:1.6; color:#d4e8f1;">The predicted class is shown as Good, Fair, Moderate, Poor, or Very Poor, along with a plain-language explanation. This project uses historical data and does not provide live sensor readings.</p>
    </div>
    """)

# -------------------- Footer --------------------
render_html("""
<div class="footer">
  🍃 &nbsp; AI Air Quality Detection &nbsp;&nbsp; | &nbsp;&nbsp;
  Powered by Machine Learning &nbsp;&nbsp; | &nbsp;&nbsp;
  For a Cleaner, Healthier Future
</div>
""")
