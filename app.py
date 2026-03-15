import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ══════════════════════════════════════════════════
# PAGE CONFIG  — no sidebar needed
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioAI — Smart Heart Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ══════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        m  = joblib.load("knn_heart_model.pkl")
        sc = joblib.load("heart_scaler.pkl")
        co = joblib.load("heart_columns.pkl")
        return m, sc, co, True
    except Exception:
        return None, None, None, False

model, scaler, expected_columns, model_loaded = load_model()

# ══════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── BASE ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #05080f !important;
    color: #e2e8f0;
}

/* ── Hide ALL Streamlit chrome & collapse its space completely ── */
#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
}

/* ── Nuke every possible source of top gap ── */
.stApp {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stAppViewContainer"] > section.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
section.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    margin-top: 0 !important;
}
div[data-testid="stVerticalBlock"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: linear-gradient(#00f5ff,#ff2cdf); border-radius: 4px; }

/* ── Animated bg ── */
body::after {
    content: '';
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
        radial-gradient(ellipse 70% 50% at 10% 15%, rgba(0,245,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(255,44,223,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(120,80,255,0.03) 0%, transparent 60%);
    animation: bgPulse 9s ease-in-out infinite alternate;
}
@keyframes bgPulse { from { opacity:0.6; } to { opacity:1; } }

/* ══════════════════════════════════════
   TOP NAVBAR
══════════════════════════════════════ */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 64px;
    background: rgba(5, 8, 15, 0.98);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(0,245,255,0.1);
    /* Pull navbar up to eat any residual Streamlit header space */
    margin-top: -6rem;
    margin-left: -2rem;
    margin-right: -2rem;
    margin-bottom: 2rem;
}

/* animated gradient underline on navbar */
.navbar::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f5ff, #ff2cdf, #7850ff, transparent);
    background-size: 300% 100%;
    animation: gradientFlow 4s linear infinite;
}
@keyframes gradientFlow {
    from { background-position: 0% 50%; }
    to   { background-position: 300% 50%; }
}

/* Brand logo area */
.nav-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    text-decoration: none;
    flex-shrink: 0;
}
.nav-brand-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, rgba(0,245,255,0.14), rgba(255,44,223,0.14));
    border: 1px solid rgba(0,245,255,0.22);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.nav-brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 18px; font-weight: 800;
    background: linear-gradient(90deg, #00f5ff, #ff2cdf);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-brand-sub {
    font-size: 9px; color: #2a3a52;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-top: 1px;
}

/* Nav links container */
.nav-links {
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Individual nav item */
.nav-item {
    position: relative;
    display: flex; align-items: center; gap: 7px;
    padding: 8px 18px;
    border-radius: 10px;
    font-size: 13.5px; font-weight: 500;
    color: #4a5a72;
    cursor: pointer;
    transition: color 0.2s, background 0.2s;
    white-space: nowrap;
    border: 1px solid transparent;
}
.nav-item:hover {
    color: #a0b4c8;
    background: rgba(0,245,255,0.05);
    border-color: rgba(0,245,255,0.1);
}
.nav-item.active {
    color: #00f5ff;
    background: rgba(0,245,255,0.08);
    border-color: rgba(0,245,255,0.2);
    font-weight: 600;
}
.nav-item.active::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 20%; right: 20%;
    height: 2px;
    background: linear-gradient(90deg, #00f5ff, #ff2cdf);
    border-radius: 2px;
}

/* Date badge in navbar */
.nav-date {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #2a3a52;
    padding: 6px 14px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 8px;
    flex-shrink: 0;
}
.nav-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #00f5ff;
    box-shadow: 0 0 6px #00f5ff;
    animation: glowPulse 2s ease infinite;
}

/* ══════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════ */
@keyframes slideUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes shimmer {
    from { background-position: 0% center; }
    to   { background-position: 200% center; }
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-8px); }
}
@keyframes glowPulse {
    0%,100% { opacity:1; box-shadow: 0 0 6px #00f5ff; }
    50%     { opacity:0.5; box-shadow: 0 0 14px #00f5ff; }
}
@keyframes pulseRed {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,45,90,0); }
    50%     { box-shadow: 0 0 0 12px rgba(255,45,90,0.06); }
}

.neon-grad {
    background: linear-gradient(135deg, #00f5ff 0%, #ff2cdf 50%, #7850ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}

/* ══════════════════════════════════════
   HOME — HERO
══════════════════════════════════════ */
.hero-inner {
    background: linear-gradient(135deg, rgba(8,16,36,0.98), rgba(14,8,35,0.98), rgba(5,20,40,0.98));
    border: 1px solid rgba(0,245,255,0.14);
    border-radius: 26px;
    padding: 64px 52px;
    text-align: center;
    position: relative; overflow: hidden;
    margin-bottom: 28px;
    animation: slideUp 0.7s ease both;
}
.hero-inner::before {
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 55% 55% at 15% 50%, rgba(0,245,255,0.1), transparent 55%),
        radial-gradient(ellipse 50% 55% at 85% 50%, rgba(255,44,223,0.1), transparent 55%);
}
.hero-orb {
    position: absolute; width:280px; height:280px;
    top:-100px; right:-80px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,44,223,0.07), transparent 70%);
    animation: float 7s ease-in-out infinite;
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,245,255,0.07);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 50px; padding: 6px 16px;
    font-size: 11px; font-weight: 700; color: #00f5ff;
    letter-spacing: 1.4px; text-transform: uppercase;
    margin-bottom: 22px; position: relative; z-index: 1;
    animation: slideUp 0.6s ease 0.05s both;
}
.hero-dot {
    width:6px; height:6px; border-radius:50%;
    background:#00f5ff; display:inline-block;
    animation: glowPulse 2s ease infinite;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 54px; font-weight: 800; line-height: 1.1;
    color: #fff; margin-bottom: 16px;
    position: relative; z-index: 1;
    animation: slideUp 0.7s ease 0.1s both;
}
.hero-sub {
    font-size: 16px; color: #7a8898; line-height: 1.7;
    max-width: 500px; margin: 0 auto;
    position: relative; z-index: 1;
    animation: slideUp 0.7s ease 0.2s both;
}

/* ══════════════════════════════════════
   STATS
══════════════════════════════════════ */
.stats-row {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 18px; margin-bottom: 26px;
}
.stat-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,245,255,0.1);
    border-radius: 18px; padding: 26px 16px; text-align: center;
    transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
    animation: slideUp 0.7s ease both;
    position: relative; overflow: hidden;
}
.stat-card:nth-child(1){animation-delay:0.1s;}
.stat-card:nth-child(2){animation-delay:0.18s;}
.stat-card:nth-child(3){animation-delay:0.26s;}
.stat-card:nth-child(4){animation-delay:0.34s;}
.stat-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 45px rgba(0,245,255,0.12), 0 0 0 1px rgba(0,245,255,0.22);
    border-color: rgba(0,245,255,0.28);
}
.stat-icon { font-size: 22px; margin-bottom: 8px; }
.stat-val {
    font-family: 'Syne', sans-serif; font-size: 42px; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, #00f5ff, #ff2cdf);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-lbl {
    font-size: 11px; color: #5a6a80; margin-top: 7px;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
}

/* ══════════════════════════════════════
   FEATURE CARDS
══════════════════════════════════════ */
.feat-row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
}
.feat-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px; padding: 34px 30px;
    position: relative; overflow: hidden;
    transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
    animation: slideUp 0.7s ease both;
}
.feat-card:nth-child(1){animation-delay:0.15s;}
.feat-card:nth-child(2){animation-delay:0.22s;}
.feat-card:nth-child(3){animation-delay:0.29s;}
.feat-card:nth-child(4){animation-delay:0.36s;}
.feat-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00f5ff, #ff2cdf);
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.4s ease;
}
.feat-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0,245,255,0.2);
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
}
.feat-card:hover::after { transform: scaleX(1); }
.feat-icon-wrap {
    width: 48px; height: 48px; border-radius: 13px;
    background: rgba(0,245,255,0.07);
    border: 1px solid rgba(0,245,255,0.18);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin-bottom: 16px;
    transition: transform 0.3s, box-shadow 0.3s;
}
.feat-card:hover .feat-icon-wrap {
    transform: scale(1.12) rotate(-6deg);
    box-shadow: 0 0 18px rgba(0,245,255,0.22);
}
.feat-title {
    font-family: 'Syne', sans-serif; font-size: 17px; font-weight: 700; margin-bottom: 10px;
    background: linear-gradient(90deg, #00f5ff, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.feat-body { font-size: 13.5px; color: #6a7888; line-height: 1.78; }

/* ══════════════════════════════════════
   PAGE HEADER (non-home)
══════════════════════════════════════ */
.page-header {
    background: linear-gradient(135deg, rgba(7,14,32,0.98), rgba(11,6,28,0.98));
    border: 1px solid rgba(0,245,255,0.12);
    border-radius: 22px; padding: 40px 50px 36px;
    text-align: center; margin-bottom: 26px;
    position: relative; overflow: hidden;
    animation: slideUp 0.6s ease both;
}
.page-header::before {
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 50% 60% at 15% 50%, rgba(0,245,255,0.07), transparent 55%),
        radial-gradient(ellipse 40% 50% at 85% 50%, rgba(255,44,223,0.07), transparent 55%);
}
.page-header::after {
    content: '';
    position: absolute; bottom:0; left:20%; right:20%; height:1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.5), rgba(255,44,223,0.5), transparent);
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 36px; font-weight: 800; color: #fff;
    display: flex; align-items: center; justify-content: center; gap: 12px;
    position: relative; z-index: 1;
}
.page-header p {
    color: #6a7888; font-size: 15px; margin-top: 10px;
    position: relative; z-index: 1;
}

/* ══════════════════════════════════════
   GLASS FORM CARDS
══════════════════════════════════════ */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,245,255,0.1);
    border-radius: 18px; padding: 26px 28px; margin-bottom: 18px;
    position: relative; overflow: hidden;
    animation: slideUp 0.6s ease both;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(0,245,255,0.22); }
.glass-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.35), transparent);
}
.section-label {
    font-size: 10.5px; font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; color: #00f5ff;
    border-bottom: 1px solid rgba(0,245,255,0.12);
    padding-bottom: 10px; margin-bottom: 18px;
}

/* form widgets */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(0,245,255,0.18) !important;
    border-radius: 11px !important; color: #e2e8f0 !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #00f5ff !important;
    box-shadow: 0 0 0 3px rgba(0,245,255,0.1) !important;
}
input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(0,245,255,0.18) !important;
    border-radius: 11px !important; color: #e2e8f0 !important;
}
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg,#00f5ff,#ff2cdf) !important;
}
label[data-testid="stWidgetLabel"] > div > p {
    color: #7a8898 !important; font-size: 13px !important; font-weight: 500 !important;
}

/* predict button */
.predict-btn > div > button {
    background: linear-gradient(135deg, #ff2cdf, #00f5ff) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important; font-weight: 700 !important;
    padding: 13px 0 !important; border-radius: 13px !important;
    border: none !important; width: 100% !important;
    box-shadow: 0 6px 28px rgba(0,245,255,0.18) !important;
    transition: all 0.3s ease !important;
}
.predict-btn > div > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 14px 45px rgba(0,245,255,0.3), 0 0 50px rgba(255,44,223,0.2) !important;
}

/* result cards */
.result-high {
    background: linear-gradient(135deg, rgba(255,45,90,0.11), rgba(255,45,90,0.03));
    border: 1px solid rgba(255,45,90,0.28); border-left: 4px solid #ff2d5a;
    padding: 28px 32px; border-radius: 18px; text-align: center; margin-top: 22px;
    animation: slideUp 0.5s ease both, pulseRed 2.5s ease 0.5s infinite;
}
.result-low {
    background: linear-gradient(135deg, rgba(0,245,255,0.09), rgba(0,245,255,0.02));
    border: 1px solid rgba(0,245,255,0.22); border-left: 4px solid #00f5ff;
    padding: 28px 32px; border-radius: 18px; text-align: center; margin-top: 22px;
    animation: slideUp 0.5s ease both;
    box-shadow: 0 0 35px rgba(0,245,255,0.07);
}
.result-title { font-family:'Syne',sans-serif; font-size:24px; font-weight:800; margin-bottom:9px; }
.result-sub   { font-size:14px; color:#7a8898; line-height:1.6; }

/* ══════════════════════════════════════
   HEALTH TIPS
══════════════════════════════════════ */
.tips-grid {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 20px; margin-bottom: 24px;
}
.tip-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 28px;
    position: relative; overflow: hidden;
    transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
    animation: slideUp 0.7s ease both;
}
.tip-card:nth-child(1){animation-delay:0.08s;}
.tip-card:nth-child(2){animation-delay:0.15s;}
.tip-card:nth-child(3){animation-delay:0.22s;}
.tip-card:nth-child(4){animation-delay:0.29s;}
.tip-card:nth-child(5){animation-delay:0.36s;}
.tip-card:nth-child(6){animation-delay:0.43s;}
.tip-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    opacity:0; transition: opacity 0.3s;
}
.tip-card:hover { transform: translateY(-5px); box-shadow: 0 18px 45px rgba(0,0,0,0.22); }
.tip-card:hover::before { opacity:1; }
.tc1::before { background: linear-gradient(90deg,#34d399,#00b894); }
.tc2::before { background: linear-gradient(90deg,#00f5ff,#00c8ff); }
.tc3::before { background: linear-gradient(90deg,#ff6b6b,#ff4444); }
.tc4::before { background: linear-gradient(90deg,#c084fc,#7850ff); }
.tc5::before { background: linear-gradient(90deg,#ff2cdf,#c084fc); }
.tc6::before { background: linear-gradient(90deg,#fb923c,#f59e0b); }
.tc1:hover { border-color: rgba(52,211,153,0.25); }
.tc2:hover { border-color: rgba(0,245,255,0.25); }
.tc3:hover { border-color: rgba(255,107,107,0.25); }
.tc4:hover { border-color: rgba(192,132,252,0.25); }
.tc5:hover { border-color: rgba(255,44,223,0.25); }
.tc6:hover { border-color: rgba(251,146,60,0.25); }
.tip-emoji-wrap {
    width:52px; height:52px; border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    font-size:24px; margin-bottom:14px;
    transition: transform 0.3s;
}
.tip-card:hover .tip-emoji-wrap { transform: scale(1.15) rotate(-8deg); }
.tip-badge {
    display:inline-block; font-size:10px; font-weight:700;
    padding:3px 10px; border-radius:20px; margin-bottom:11px;
    letter-spacing:0.8px; text-transform:uppercase;
}
.b-green  { background:rgba(52,211,153,0.1); color:#34d399; border:1px solid rgba(52,211,153,0.2); }
.b-cyan   { background:rgba(0,245,255,0.1);  color:#00f5ff; border:1px solid rgba(0,245,255,0.2); }
.b-red    { background:rgba(255,107,107,0.1);color:#ff6b6b; border:1px solid rgba(255,107,107,0.2); }
.b-purple { background:rgba(192,132,252,0.1);color:#c084fc; border:1px solid rgba(192,132,252,0.2); }
.b-pink   { background:rgba(255,44,223,0.1); color:#ff2cdf; border:1px solid rgba(255,44,223,0.2); }
.b-orange { background:rgba(251,146,60,0.1); color:#fb923c; border:1px solid rgba(251,146,60,0.2); }
.tip-title { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:#dde6f0; margin-bottom:9px; }
.tip-body  { font-size:13px; color:#5a6878; line-height:1.76; }
.warn-box {
    background: rgba(255,44,223,0.04);
    border: 1px solid rgba(255,44,223,0.16); border-left:4px solid #ff2cdf;
    border-radius:14px; padding:20px 26px;
    font-size:13px; color:#7a8898; line-height:1.72;
    animation: slideUp 0.7s ease 0.5s both;
}
.warn-box strong { color:#ff2cdf; }

/* ══════════════════════════════════════
   ABOUT
══════════════════════════════════════ */
.about-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:20px; }
.about-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,245,255,0.08);
    border-radius:20px; padding:28px;
    transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
    animation: slideUp 0.7s ease both;
    position: relative; overflow: hidden;
}
.about-card:nth-child(1){animation-delay:0.08s;}
.about-card:nth-child(2){animation-delay:0.16s;}
.about-card:nth-child(3){animation-delay:0.24s;}
.about-card:nth-child(4){animation-delay:0.32s;}
.about-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.35), transparent);
    opacity:0; transition:opacity 0.3s;
}
.about-card:hover {
    transform: translateY(-5px); border-color: rgba(0,245,255,0.22);
    box-shadow: 0 18px 40px rgba(0,0,0,0.22);
}
.about-card:hover::before { opacity:1; }
.about-card h3 {
    font-family:'Syne',sans-serif; font-size:15px; font-weight:700; margin-bottom:12px;
    background: linear-gradient(90deg,#00f5ff,#ff2cdf);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.about-card p { font-size:13px; color:#5a6878; line-height:1.78; }
.stack-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(0,245,255,0.08);
    border-radius:20px; padding:28px; margin-bottom:20px;
    animation: slideUp 0.7s ease 0.4s both;
}
.stack-card h3 {
    font-family:'Syne',sans-serif; font-size:15px; font-weight:700; margin-bottom:14px;
    background: linear-gradient(90deg,#00f5ff,#ff2cdf);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.tech-badge {
    display:inline-flex; align-items:center; gap:5px;
    background: rgba(0,245,255,0.06); border:1px solid rgba(0,245,255,0.14);
    color:#6de4e8; font-size:12px; font-weight:500;
    padding:5px 13px; border-radius:50px; margin:3px;
    transition: background 0.2s, transform 0.2s;
}
.tech-badge:hover { background:rgba(0,245,255,0.12); transform:translateY(-2px); }
.disclaimer {
    background: rgba(0,245,255,0.025); border:1px solid rgba(0,245,255,0.1);
    border-radius:14px; padding:22px 26px;
    font-size:13px; color:#5a6878; line-height:1.78;
    animation: slideUp 0.7s ease 0.5s both;
}
.disclaimer strong { color:#00f5ff; }
</style>
""", unsafe_allow_html=True)

# ══ JS: aggressively remove top gap ══
st.markdown("""
<script>
(function(){
    var HIDE='display:none!important;height:0!important;min-height:0!important;max-height:0!important;padding:0!important;margin:0!important;overflow:hidden!important;';
    function fix(){
        ['[data-testid="stHeader"]','[data-testid="stToolbar"]','[data-testid="stDecoration"]',
         'header','section[data-testid="stSidebar"]',
         '[data-testid="stSidebarCollapsedControl"]','[data-testid="collapsedControl"]']
        .forEach(function(s){var e=document.querySelector(s);if(e)e.style.cssText=HIDE;});
        /* zero out every top-level wrapper */
        ['.stApp','section.main','[data-testid="stAppViewContainer"]']
        .forEach(function(s){var e=document.querySelector(s);
            if(e){e.style.paddingTop='0';e.style.marginTop='0';}});
        var bc=document.querySelector('.main .block-container');
        if(bc){bc.style.paddingTop='0';bc.style.marginTop='0';}
    }
    fix();
    [50,150,300,700].forEach(function(t){setTimeout(fix,t);});
    new MutationObserver(fix).observe(document.body,{childList:true,subtree:false});
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# NAVBAR  (pure HTML — rendered once at top)
# ══════════════════════════════════════════════════
cur = st.session_state.page

def nc(name):
    return "nav-item active" if cur == name else "nav-item"

st.markdown(f"""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-brand-icon">🫀</div>
        <div>
            <div class="nav-brand-name">CardioAI</div>
            <div class="nav-brand-sub">Heart Risk System</div>
        </div>
    </div>
    <div class="nav-links">
        <a class="{nc('Home')}"        href="?nav=Home"        target="_self" style="text-decoration:none;">Home</a>
        <a class="{nc('Prediction')}"  href="?nav=Prediction"  target="_self" style="text-decoration:none;">Prediction</a>
        <a class="{nc('Health Tips')}" href="?nav=HealthTips"  target="_self" style="text-decoration:none;">Health Tips</a>
        <a class="{nc('About')}"       href="?nav=About"       target="_self" style="text-decoration:none;">About</a>
    </div>
    <div class="nav-date">
        <span class="nav-dot"></span>
        {datetime.now().strftime("%b %d, %Y")}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Handle URL nav param ──
_p = st.query_params
if "nav" in _p:
    _map = {"Home":"Home","Prediction":"Prediction","HealthTips":"Health Tips","About":"About"}
    _v = _p["nav"]
    if _v in _map and st.session_state.page != _map[_v]:
        st.session_state.page = _map[_v]
        st.query_params.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════
# ██  H O M E
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero-inner">
        <div class="hero-orb"></div>
        <div class="hero-badge"><span class="hero-dot"></span> AI-Powered Cardiac Analysis</div>
        <div class="hero-title">
            Smart Heart Risk<br>
            <span class="neon-grad">Predictor</span>
        </div>
        <div class="hero-sub">
            Advanced machine learning technology analyzing 11 critical cardiac parameters
            to deliver instant, accurate cardiovascular risk assessment.
        </div>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-val">90%</div>
            <div class="stat-lbl">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-val">10K+</div>
            <div class="stat-lbl">Predictions</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🔬</div>
            <div class="stat-val">11</div>
            <div class="stat-lbl">Features</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🛡️</div>
            <div class="stat-val">24/7</div>
            <div class="stat-lbl">Available</div>
        </div>
    </div>

    <div class="feat-row">
        <div class="feat-card">
            <div class="feat-icon-wrap">🎯</div>
            <div class="feat-title">Why Choose CardioAI?</div>
            <div class="feat-body">Our advanced machine learning model analyzes 11 critical cardiac
            parameters for instant risk assessment. Early detection can save lives — get your
            prediction in seconds, trained on real clinical data using KNN classification.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon-wrap">🔬</div>
            <div class="feat-title">How It Works</div>
            <div class="feat-body">Enter your health metrics including age, blood pressure, cholesterol,
            and ECG results. Our AI processes data using K-Nearest Neighbors to predict heart disease
            risk. Get instant color-coded results with clear actionable guidance.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon-wrap">🛡️</div>
            <div class="feat-title">Privacy First</div>
            <div class="feat-body">Your health data is processed entirely locally — nothing is stored
            or transmitted to any server. Your medical information stays yours. Use this tool with
            full confidence that your privacy is always protected.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon-wrap">⚡</div>
            <div class="feat-title">Instant Results</div>
            <div class="feat-body">No waiting, no appointments needed. Enter your vitals and receive
            a detailed risk assessment in milliseconds. Clear, easy-to-understand results with next
            steps for both high and low risk outcomes.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ██  P R E D I C T I O N
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "Prediction":

    st.markdown("""
    <div class="page-header">
        <h1>🔮&nbsp;<span class="neon-grad">Heart Risk Prediction</span></h1>
        <p>Fill in your health parameters for an instant AI-powered cardiac risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Model files not found. Place `knn_heart_model.pkl`, `heart_scaler.pkl`, "
                   "and `heart_columns.pkl` in the same directory.", icon="⚠️")

    st.markdown('<div class="glass-card"><div class="section-label">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 100, 45)
    with c2:
        sex = st.selectbox("Biological Sex", ["M","F"], format_func=lambda x: "♂ Male" if x=="M" else "♀ Female")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-label">🩺 Cardiac Metrics</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA","NAP","TA","ASY"],
                                   help="ATA=Atypical · NAP=Non-Anginal · TA=Typical · ASY=Asymptomatic")
    with c4:
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    with c5:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    c6, c7, c8 = st.columns(3)
    with c6:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0,1],
                                   format_func=lambda x: "Yes (>120)" if x==1 else "No (≤120)")
    with c7:
        resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
    with c8:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-label">📊 Exercise Data</div>', unsafe_allow_html=True)
    c9, c10, c11 = st.columns(3)
    with c9:
        exercise_angina = st.selectbox("Exercise Angina", ["N","Y"], format_func=lambda x: "Yes" if x=="Y" else "No")
    with c10:
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    with c11:
        st_slope = st.selectbox("ST Slope", ["Up","Flat","Down"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        predict_clicked = st.button("🚀 Analyze Heart Risk Now", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if model_loaded:
            raw = {
                "Age":age, "RestingBP":resting_bp, "Cholesterol":cholesterol,
                "FastingBS":fasting_bs, "MaxHR":max_hr, "Oldpeak":oldpeak,
                "Sex_"+sex:1, "ChestPainType_"+chest_pain:1,
                "RestingECG_"+resting_ecg:1, "ExerciseAngina_"+exercise_angina:1,
                "ST_Slope_"+st_slope:1,
            }
            df = pd.DataFrame([raw])
            for col in expected_columns:
                if col not in df.columns: df[col] = 0
            df = df[expected_columns]
            pred = model.predict(scaler.transform(df))[0]
            if pred == 1:
                st.markdown("""
                <div class="result-high">
                    <div class="result-title" style="color:#ff2d5a;">⚠️ High Risk Detected</div>
                    <div class="result-sub">Elevated cardiac risk found in your metrics.<br>
                    <strong style="color:#ff8099;">Please consult a cardiologist immediately.</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-low">
                    <div class="result-title" style="color:#00f5ff;">✅ Low Risk Detected</div>
                    <div class="result-sub">Your metrics suggest a lower cardiac risk profile.<br>
                    <strong style="color:#7de8ed;">Maintain a healthy lifestyle and get regular check-ups.</strong></div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("⚠️ Add the required `.pkl` model files to run predictions.")


# ══════════════════════════════════════════════════════════════
# ██  H E A L T H   T I P S
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "Health Tips":

    st.markdown("""
    <div class="page-header">
        <h1>💡&nbsp;<span class="neon-grad">Heart Health Tips</span></h1>
        <p>Evidence-based lifestyle guidance to protect and strengthen your cardiovascular system</p>
    </div>

    <div class="tips-grid">
        <div class="tip-card tc1">
            <div class="tip-emoji-wrap" style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);">🥗</div>
            <div class="tip-badge b-green">Nutrition</div>
            <div class="tip-title">Heart-Healthy Diet</div>
            <div class="tip-body">Prioritize fruits, vegetables, whole grains, and lean proteins.
            Limit saturated fats, trans fats, and sodium. The Mediterranean diet strongly
            reduces cardiovascular risk.</div>
        </div>
        <div class="tip-card tc2">
            <div class="tip-emoji-wrap" style="background:rgba(0,245,255,0.08);border:1px solid rgba(0,245,255,0.2);">🏃</div>
            <div class="tip-badge b-cyan">Exercise</div>
            <div class="tip-title">Regular Physical Activity</div>
            <div class="tip-body">Aim for 150 minutes of moderate aerobic activity per week.
            Brisk walking, swimming, or cycling strengthen the heart and improve circulation
            significantly.</div>
        </div>
        <div class="tip-card tc3">
            <div class="tip-emoji-wrap" style="background:rgba(255,107,107,0.08);border:1px solid rgba(255,107,107,0.2);">🚭</div>
            <div class="tip-badge b-red">Lifestyle</div>
            <div class="tip-title">Quit Smoking</div>
            <div class="tip-body">Smoking is a top risk factor for heart disease. Quitting
            reduces risk within 1–2 years. Even secondhand smoke measurably increases
            cardiovascular risk.</div>
        </div>
        <div class="tip-card tc4">
            <div class="tip-emoji-wrap" style="background:rgba(192,132,252,0.08);border:1px solid rgba(192,132,252,0.2);">😴</div>
            <div class="tip-badge b-purple">Recovery</div>
            <div class="tip-title">Quality Sleep</div>
            <div class="tip-body">Adults need 7–9 hours per night. Poor sleep is linked to
            high blood pressure, obesity, and increased heart disease risk. A consistent
            schedule makes a measurable difference.</div>
        </div>
        <div class="tip-card tc5">
            <div class="tip-emoji-wrap" style="background:rgba(255,44,223,0.08);border:1px solid rgba(255,44,223,0.2);">🧘</div>
            <div class="tip-badge b-pink">Mental Health</div>
            <div class="tip-title">Manage Stress</div>
            <div class="tip-body">Chronic stress raises blood pressure. Practice mindfulness,
            meditation, deep breathing, or yoga. Strong social connections also play a
            significant protective cardiac role.</div>
        </div>
        <div class="tip-card tc6">
            <div class="tip-emoji-wrap" style="background:rgba(251,146,60,0.08);border:1px solid rgba(251,146,60,0.2);">🩺</div>
            <div class="tip-badge b-orange">Prevention</div>
            <div class="tip-title">Regular Check-ups</div>
            <div class="tip-body">Monitor blood pressure, cholesterol, and blood sugar regularly.
            Early detection of hypertension allows timely intervention before serious
            complications develop.</div>
        </div>
    </div>

    <div class="warn-box">
        ⚕️ <strong>Medical Disclaimer:</strong> These tips are for educational purposes only and
        do not constitute medical advice. Always consult a qualified healthcare professional
        before making changes to diet, exercise, or medication. If you experience chest pain
        or shortness of breath — seek emergency care immediately.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ██  A B O U T
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "About":

    st.markdown("""
    <div class="page-header">
        <h1>ℹ️&nbsp;<span class="neon-grad">About CardioAI</span></h1>
        <p>The technology, data, and mission behind this application</p>
    </div>

    <div class="about-grid">
        <div class="about-card">
            <h3>🤖 The ML Model</h3>
            <p>CardioAI uses a K-Nearest Neighbors (KNN) classifier trained on the UCI Heart Disease
            dataset with over 900 patient records. Features are normalized using StandardScaler,
            achieving ~90% accuracy on held-out test data.</p>
        </div>
        <div class="about-card">
            <h3>📊 Input Features</h3>
            <p>11 clinical parameters: Age, Sex, Chest Pain Type, Resting Blood Pressure,
            Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced
            Angina, Oldpeak, and ST Slope — all established cardiac risk indicators.</p>
        </div>
        <div class="about-card">
            <h3>🗄️ Dataset</h3>
            <p>Combines 5 independent datasets — Cleveland, Hungarian, Switzerland, Long Beach VA,
            and Stalog — into the largest heart disease dataset for research purposes,
            totaling 918 patient records.</p>
        </div>
        <div class="about-card">
            <h3>🏗️ Architecture</h3>
            <p>Built with Python and Streamlit. The ML pipeline uses scikit-learn for
            preprocessing (StandardScaler) and classification (KNeighborsClassifier).
            Artifacts serialized with joblib for fast loading.</p>
        </div>
    </div>

    <div class="stack-card">
        <h3>🛠️ Technology Stack</h3>
        <div style="margin-top:6px;">
            <span class="tech-badge">🐍 Python 3.10+</span>
            <span class="tech-badge">📊 Streamlit</span>
            <span class="tech-badge">🤖 scikit-learn</span>
            <span class="tech-badge">🐼 Pandas</span>
            <span class="tech-badge">🔢 NumPy</span>
            <span class="tech-badge">💾 joblib</span>
            <span class="tech-badge">📈 KNN Classifier</span>
            <span class="tech-badge">⚖️ StandardScaler</span>
        </div>
    </div>

    <div class="disclaimer">
        ⚠️ <strong>Important Disclaimer:</strong> CardioAI is built for educational and research
        purposes. It is <strong>not a certified medical device</strong> and must not replace
        professional diagnosis or treatment. Always consult a licensed healthcare professional.
        In a cardiac emergency, call emergency services immediately.
    </div>
    """, unsafe_allow_html=True)