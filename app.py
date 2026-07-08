import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Setup Page Configuration
st.set_page_config(
    page_title="Bankruptcy Prediction System",
    page_icon="Favicon icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (Logo moved above navigation bar)

# Custom CSS for improved UI Aesthetics
st.markdown("""
<style>
    @keyframes pageLoad {
        0% { opacity: 0; transform: scale(0.98) translateY(10px); filter: blur(2px); }
        100% { opacity: 1; transform: scale(1) translateY(0); filter: blur(0px); }
    }
    .block-container {
        padding-top: 2rem;
        animation: pageLoad 0.5s cubic-bezier(0.25, 1, 0.5, 1) forwards;
    }
    .stButton>button {
        width: 100%;
        background-color: #174F77;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #4D8CB5;
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .highlight {
        background-color: #94BED8;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #2A1A31;
        transition: all 0.3s ease-in-out;
    }
    .highlight:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] {
        color: #174F77;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stMetricValue"]:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        rf = joblib.load("rf_model.pkl")
        ann = joblib.load("ann_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return rf, ann, scaler
    except Exception as e:
        return None, None, None

rf_model, ann_model, scaler = load_models()

# --- Header Logo & Navigation ---
header_col1, header_col2 = st.columns([1, 6], gap="large", vertical_alignment="center")

with header_col1:
    st.image("Favicon icon.png", use_container_width=True)

with header_col2:
    # --- Top Navigation Bar ---
    page = option_menu(
        menu_title=None,
        options=["Overview", "Model Analysis", "About"],
        icons=["house", "graph-up", "person"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "8px!important", 
                "background-color": "#D9C9B7", 
                "border-radius": "50px",
                "box-shadow": "0 4px 10px rgba(0,0,0,0.1)",
                "margin": "0 0 0 auto",
                "max-width": "600px",
                "float": "right"
            },
            "icon": {"color": "#174F77", "font-size": "18px", "margin-right": "8px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "center", 
                "margin": "0px 5px", 
                "--hover-color": "#94BED8", 
                "color": "#2A1A31",
                "border-radius": "50px",
                "padding": "10px 20px",
                "transition": "all 0.3s ease-in-out"
            },
            "nav-link-selected": {
                "background-color": "#174F77", 
                "color": "white",
                "border-radius": "50px",
                "font-weight": "600",
                "box-shadow": "0 4px 6px rgba(0,0,0,0.15)"
            },
        }
    )
st.sidebar.info("Developed by: **Syed Muhammad Saad Hussain Zaidi**")

# --- Overview Page ---
if page == "Overview":
    st.title("Financial Bankruptcy Prediction System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Bankruptcy?
        Bankruptcy is a legal proceeding involving a person or business that is unable to repay their outstanding debts. 
        For corporations, declaring bankruptcy usually means that the company’s liabilities exceed its assets, and it can no longer operate successfully.
        
        ### Why is this Web Application Important?
        Predicting bankruptcy before it happens is extremely critical for several stakeholders:
        - **Investors:** To protect their capital from being lost in failing businesses.
        - **Creditors & Banks:** To assess credit risk before issuing loans.
        - **Management:** To identify financial distress early and take corrective actions.
        
        This system utilizes advanced **Machine Learning** algorithms, including a standalone **Artificial Neural Network (ANN)** and an **Ensemble Hybrid Model** (Random Forest + ANN), to analyze key macroeconomic and company-specific financial indicators to predict the risk of default.
        """)
        
    with col2:
        st.info("**Key Economic Indicators Tracked:**\n\n"
                "1. Return on Asset\n"
                "2. Asset Turnover\n"
                "3. Real Interest Rate\n"
                "4. GDP Growth Annual\n"
                "5. Electricity Prices")
        
        lottie_finance = load_lottieurl("https://lottie.host/9e1e9ff5-58fa-4e7d-9442-5ba0a109a250/ZONh2Hj4a7.json")
        if lottie_finance:
            st_lottie(lottie_finance, height=200, key="finance")
            
    st.markdown("---")
    st.subheader("How to Use")
    st.markdown("""
    1. Navigate to the **Model Analysis** page using the sidebar on the left.
    2. Select the prediction model you wish to use (ANN or Hybrid).
    3. Enter the financial indicators for the company/economy.
    4. Click **Predict Risk** to see the probability of bankruptcy.
    """)

# --- Model Analysis Page ---
elif page == "Model Analysis":
    st.title("Bankruptcy Model Analysis")
    st.markdown("Use this interface to input financial data and get real-time risk predictions.")
    
    if rf_model is None or ann_model is None or scaler is None:
        st.error("Models or scaler could not be loaded. Please ensure `rf_model.pkl`, `ann_model.pkl`, and `scaler.pkl` are present in the directory.")
    else:
        # Model Selection
        model_choice = st.radio("Choose the Predictive Model:", ["Hybrid Model (Random Forest + ANN)", "Standalone ANN Model"], horizontal=True)
        st.markdown("---")
        
        with st.form("prediction_form"):
            st.subheader("Enter Economic & Financial Indicators")
            
            col1, col2 = st.columns(2)
            with col1:
                return_on_asset = st.number_input("Return on Asset", min_value=-200.0, max_value=400.0, value=4.8, step=0.1, help="Net income divided by total assets.")
                real_interest_rate = st.number_input("Real Interest Rate (%)", min_value=-20.0, max_value=20.0, value=2.0, step=0.1, help="Interest rate adjusted for inflation.")
                electricity_prices = st.number_input("Electricity Prices", min_value=0.0, max_value=50.0, value=12.8, step=0.1, help="Cost of electricity, impacting operational overhead.")
            
            with col2:
                asset_turnover = st.number_input("Asset Turnover", min_value=0.0, max_value=30.0, value=1.1, step=0.1, help="Revenue divided by total assets.")
                gdp_growth = st.number_input("GDP Growth (Annual %)", min_value=-10.0, max_value=15.0, value=3.8, step=0.1, help="Macroeconomic annual GDP growth rate.")
                
            predict_button = st.form_submit_button("Predict Risk")

        if predict_button:
            input_data = np.array([[return_on_asset, asset_turnover, real_interest_rate, gdp_growth, electricity_prices]])
            input_scaled = scaler.transform(input_data)

            with st.spinner("Analyzing financial data..."):
                if "Hybrid" in model_choice:
                    rf_proba = rf_model.predict_proba(input_scaled)[:, 1]
                    ann_proba = ann_model.predict_proba(input_scaled)[:, 1]
                    final_proba = (rf_proba[0] + ann_proba[0]) / 2
                    model_name = "Hybrid Model"
                else:
                    final_proba = ann_model.predict_proba(input_scaled)[:, 1][0]
                    model_name = "ANN Model"

                predicted_class = int(final_proba > 0.5)

            st.markdown("---")
            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if predicted_class == 1:
                    st.error(f"**Prediction:** High Risk of Bankruptcy")
                else:
                    st.success(f"**Prediction:** Financially Stable (Not Bankrupt)")
                    
            with res_col2:
                st.metric(label=f"Bankruptcy Probability ({model_name})", value=f"{final_proba*100:.2f}%")
                
            # Progress bar visualization
            st.markdown("### Risk Meter")
            st.progress(float(final_proba))

# --- About Page ---
elif page == "About":
    st.title("About the Developer")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Developer profile image
        st.image("WhatsApp Image 2026-07-08 at 5.46.52 PM.jpeg", use_container_width=True)
    
    with col2:
        st.header("Syed Muhammad Saad Hussain Zaidi")
        st.subheader("Data Scientist | Physicist | Web Developer")
        st.markdown("""
        Hello! I am a multi-disciplinary professional bridging the gap between rigorous scientific analysis and modern software development.
        
        **My Expertise:**
        - **Physics:** Grounded in analytical thinking, problem-solving, and understanding complex systems.
        - **Data Science:** Specialized in Machine Learning, Deep Learning (ANNs), and predictive modeling to extract actionable insights from raw data.
        - **Web Development:** Capable of turning complex backend logic and models into intuitive, user-friendly web applications.
        
        **About This Project:**
        This Bankruptcy Prediction System was built to demonstrate how advanced Machine Learning (combining Random Forests and Neural Networks) can be seamlessly integrated into a web interface to solve real-world financial problems.
        """)
        
    st.markdown("---")
    st.markdown("### Connect & Collaborate")
    st.info("I am always open to discussing data science, physics, and new web technologies. Feel free to explore the models provided in this application!")
