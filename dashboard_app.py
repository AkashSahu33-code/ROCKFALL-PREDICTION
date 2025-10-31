import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="AI Rockfall Prediction", layout="wide")

# --- Animated Gradient Alert Bar CSS ---
st.markdown("""
<style>
body, .main, .block-container { background-color: #17171c !important; color: #f6f6f6 !important; }
section[data-testid="stSidebar"] { background-color: #232329 !important; }
section[data-testid="stNumberInput"] > div > div > input {
    background-color: #232329 !important;
    color: #f6f6f6 !important;
    border-radius: 8px !important;
    border: 1px solid #444 !important;
}
.section-card {
    background-color: #232329 !important;
    border-radius: 18px;
    padding: 32px 36px 24px 36px;
    margin-bottom: 38px;
    box-shadow: 0 8px 32px rgb(0 0 0 / 0.24);
}
h1 { 
    color: #40c9ff !important;
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: 0.035em; 
    margin-bottom: 48px; 
}
h2, h3 { color: #ff8cfc !important; }
hr { border-top: 1px solid #555 !important; }
.animated-alert {
    background: linear-gradient(90deg, #ff7f7f, #ff7979, #a18cd1, #fbc2eb, #fbc2eb, #8fd3f4, #84fab0, #ff7f7f);
    background-size: 600% 600%;
    animation: rainbowBar 3s ease-in-out infinite;
    padding: 23px 10px;
    border-radius: 16px;
    color: #fff;
    font-size: 1.4rem;
    font-weight: bold;
    box-shadow: 0 0 24px 4px #ff8a8a55;
    text-shadow: 0 1px 8px #80000066;
    letter-spacing: .03em;
    margin-bottom:22px;
    border: 4px solid transparent;
    border-image: linear-gradient(90deg,#ff7f7f,#b2fefa,#8fd3f4,#f7971e);
    border-image-slice: 1;
}
@keyframes rainbowBar {
    0% { background-position:0% 50% }
    50% { background-position:100% 50% }
    100% { background-position:0% 50% }
}
</style>
""", unsafe_allow_html=True)

st.title("AI Rockfall Prediction System")

with st.sidebar:
    st.header("Project & Model")
    model_accuracies = {"Random Forest": 0.92, "XGBoost": 0.94, "SVM": 0.89}
    model_choice = st.selectbox("Select Model", list(model_accuracies.keys()))
    st.markdown(f"**Model Accuracy:** {model_accuracies[model_choice]*100:.2f}%")
    st.markdown("---")

# Load models and scaler
rf = joblib.load('random_forest_model.pkl')
xgb = joblib.load('xgboost_model.pkl')
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

features = [
    'displacement', 'velocity', 'cav', 'energy', 'rainfall', 'temp',
    'crack_length', 'bench_height', 'slope_angle', 'rmr', 'joint_spacing'
]

feature_help = {
    "displacement": "Rock layer movement (mm)",
    "velocity": "Speed of displacement (mm/day)",
    "cav": "Cavitation factor (unitless)",
    "energy": "Energy (unitless, higher = more instability)",
    "rainfall": "Rainfall (mm, last 24h)",
    "temp": "Temperature (°C)",
    "crack_length": "Major crack length (m)",
    "bench_height": "Bench height (m)",
    "slope_angle": "Slope angle (deg)",
    "rmr": "Rock Mass Rating (0-100, lower=worse)",
    "joint_spacing": "Joint spacing (m)"
}

tabs = st.tabs(["Prediction", "Feature Importance", "About / Tutorial"])

with tabs[0]:
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("Enter Geological & Environmental Features")
        cols = st.columns(2)
        user_inputs = {}
        for i, f in enumerate(features):
            with cols[i % 2]:
                user_inputs[f] = st.number_input(
                    f"{f.replace('_', ' ').title()}",
                    value=0.0,
                    format="%.4f",
                    help=feature_help[f]
                )
        st.markdown('</div>', unsafe_allow_html=True)

    input_df = pd.DataFrame([user_inputs])
    input_scaled = scaler.transform(input_df)
    model = {"Random Forest": rf, "XGBoost": xgb, "SVM": svm}[model_choice]
    pred_prob = model.predict_proba(input_scaled)[0][1]
    alert_threshold = 0.5

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("Rockfall Risk Prediction")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            title={'text': "Risk Probability (%)"},
            number={'font': {'size': 50, 'color': '#40c9ff', 'weight': 'bold'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor':'#ccc'},
                'bar': {'color': "#40c9ff"},
                'bgcolor': "#232329",
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},
                    {'range': [40, 70], 'color': "#f9ca24"},
                    {'range': [70, 100], 'color': "#e84118"}
                ],
                'threshold': {
                    'line': {'color': "#e84118", 'width': 6},
                    'thickness': 0.8,
                    'value': alert_threshold * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # --- Animated Alert ---
        if pred_prob >= alert_threshold:
            st.markdown(
                '<div class="animated-alert">🚨 <b>High Rockfall Risk Detected!</b> Please take <span style="text-decoration: underline; color: #fff700;">immediate action</span>!</div>',unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="animated-alert" style="background:linear-gradient(90deg,#00C9A7,#92fe9d,#27ae60,#92fe9d);color:#232329;">✅<b> Rockfall Risk is Low.</b> Monitoring can continue safely.</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("Feature Importance")
        if model_choice in ["Random Forest", "XGBoost"]:
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": features, "Importance": importances * 100})
            feat_imp = feat_imp.sort_values("Importance", ascending=True)
            bar_colors = plt.cm.plasma(feat_imp["Importance"] / feat_imp["Importance"].max())
            fig2, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(
                feat_imp["Feature"],
                feat_imp["Importance"],
                color=bar_colors,
                edgecolor='#40c9ff',
                linewidth=2,
                alpha=0.97
            )
            ax.set_xlabel("Importance (%)", fontsize=14, color='#f6f6f6')
            ax.set_title(f"{model_choice} Feature Importance", fontsize=20, color='#f6f6f6')
            ax.set_facecolor('#232329')
            fig2.patch.set_facecolor('#232329')
            ax.tick_params(axis='x', colors='#f6f6f6')
            ax.tick_params(axis='y', colors='#f6f6f6')
            for spine in ax.spines.values():
                spine.set_color('#40c9ff')
            plt.tight_layout()
            st.pyplot(fig2)
            st.caption("Lighter bars indicate higher model influence on prediction.")
        else:
            st.info("Feature importance not available for SVM model.")
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("How Rockfall Prediction App Works")
    st.markdown("""
- **Model Selection:** Random Forest, XGBoost, SVM trained on mine dataset.
- **Inputs:** Enter geological/environmental conditions (hover ⚡ for help).
- **Prediction:** Risk is instantly calculated using the AI model.
- **Risk Gauge:** Clear color-coded reading for safety team.
- **Alert:** Big banners/warnings for high-risk.
- **Feature Importance:** See which parameters matter the most.
- **Contact:** For support, reach out to admin@sihsupport.com
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
---
<div style='text-align:center; color:#f6f6f6; font-size:14px;'>© 2025 AI Rockfall Prediction System &mdash; For SIH, NIT Raipur</div>
""", unsafe_allow_html=True)
