import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import zipfile
import os
import folium


st.set_page_config(
    page_title="InsightPlate Analytics",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Create navigation functions
def nav_to(page_name):
    st.session_state.page = page_name
    st.session_state.button_clicked = True

# Styles globaux
st.markdown("""
    <style>
    /* Variables globales */
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #3498DB;
        --accent-color: #E74C3C;
        --success-color: #2ECC71;
        --warning-color: #F1C40F;
        --background-start: #1a1c20;
        --background-end: #2C3E50;
        --text-color: #ECF0F1;
        --card-bg: rgba(255, 255, 255, 0.05);
        --glass-effect: rgba(255, 255, 255, 0.1);
    }

    /* Style global */
    .stApp {
        background: linear-gradient(135deg, var(--background-start), var(--background-end));
        color: var(--text-color);
    }

    /* Animation de fond */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, 
            rgba(52, 152, 219, 0.1),
            rgba(231, 76, 60, 0.1),
            rgba(46, 204, 113, 0.1));
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        z-index: -1;
    }

     /* Glass Card Effect */
    .glass-card {
        background: var(--glass-effect);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
            
    /* Headers */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    h1 {
        font-size: 2.5rem;
        text-align: center;
        background: linear-gradient(45deg, var(--secondary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
     /* Card Images */
    .card-image {
        width: 100%;
        height: 160px;
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        margin-bottom: 15px;
    }

    .powerbi-image {
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160"><rect width="200" height="160" fill="%232C3E50"/><rect x="40" y="100" width="20" height="40" fill="%233498DB"/><rect x="70" y="80" width="20" height="60" fill="%232980B9"/><rect x="100" y="60" width="20" height="80" fill="%233498DB"/><rect x="130" y="40" width="20" height="100" fill="%232980B9"/><path d="M40 90 Q100 30 150 20" stroke="%23E74C3C" stroke-width="3" fill="none"/><circle cx="40" cy="90" r="4" fill="%23E74C3C"/><circle cx="70" cy="70" r="4" fill="%23E74C3C"/><circle cx="100" cy="50" r="4" fill="%23E74C3C"/><circle cx="130" cy="30" r="4" fill="%23E74C3C"/><circle cx="150" cy="20" r="4" fill="%23E74C3C"/></svg>');
    }
    .visual-image {
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160"><rect width="200" height="160" fill="%232C3E50"/><rect x="30" y="90" width="20" height="50" fill="%233498DB"/><rect x="60" y="70" width="20" height="70" fill="%23E74C3C"/><rect x="90" y="50" width="20" height="90" fill="%239B59B6"/><rect x="120" y="30" width="20" height="110" fill="%23F1C40F"/><rect x="150" y="110" width="20" height="30" fill="%2349A44A"/><line x1="20" y1="140" x2="180" y2="140" stroke="%23FFFFFF" stroke-width="1"/><line x1="20" y1="10" x2="20" y2="140" stroke="%23FFFFFF" stroke-width="1"/><text x="10" y="145" fill="%23FFFFFF" font-size="8" font-family="Arial">0</text><text x="10" y="125" fill="%23FFFFFF" font-size="8" font-family="Arial">20</text><text x="10" y="85" fill="%23FFFFFF" font-size="8" font-family="Arial">60</text><text x="10" y="45" fill="%23FFFFFF" font-size="8" font-family="Arial">100</text><text x="10" y="15" fill="%23FFFFFF" font-size="8" font-family="Arial">140</text></svg>');

    }

    .ml-image {
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160"><rect width="200" height="160" fill="%232C3E50"/><circle cx="40" cy="40" r="10" fill="%239B59B6"/><circle cx="40" cy="80" r="10" fill="%239B59B6"/><circle cx="40" cy="120" r="10" fill="%239B59B6"/><circle cx="100" cy="60" r="10" fill="%238E44AD"/><circle cx="100" cy="100" r="10" fill="%238E44AD"/><circle cx="160" cy="80" r="10" fill="%239B59B6"/><path d="M50 40 L90 60" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 40 L90 100" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 80 L90 60" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 80 L90 100" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 120 L90 60" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 120 L90 100" stroke="%23ECF0F1" stroke-width="2"/><path d="M110 60 L150 80" stroke="%23ECF0F1" stroke-width="2"/><path d="M110 100 L150 80" stroke="%23ECF0F1" stroke-width="2"/><path d="M170 80 L190 80" stroke="%23E74C3C" stroke-width="3"/><path d="M185 75 L190 80 L185 85" fill="none" stroke="%23E74C3C" stroke-width="3"/></svg>');
    }
            
    /* Navigation */
  .nav-button {
    background: linear-gradient(135deg, rgba(85, 85, 255, 0.7), rgba(0, 153, 204, 0.7)); /* Gradient effect */
    border: 1px solid rgba(255, 255, 255, 0.3); /* Subtle border for depth */
    color: #fff; /* White text for contrast */
    padding: 1rem 2rem; /* Spacious padding */
    border-radius: 12px; /* Rounded corners */
    cursor: pointer; /* Pointer cursor for interactivity */
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease; /* Smooth transitions */
    width: 100%; /* Full-width button */
    margin: 0.5rem 0; /* Spacing between buttons */
    font-weight: 600; /* Slightly bolder text */
    font-size: 1rem; /* Adjust font size */
    box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2); /* Add a shadow for elevation */
    backdrop-filter: blur(10px); /* Glassmorphism effect */
    text-align: center; /* Center the text */
    text-transform: uppercase; /* Make text uppercase */
}

.nav-button:hover {
    transform: translateY(-3px); /* Slight lift on hover */
    box-shadow: 0px 12px 20px rgba(0, 0, 0, 0.3); /* Enhanced shadow on hover */
    background: linear-gradient(135deg, rgba(0, 153, 204, 0.8), rgba(85, 85, 255, 0.8)); /* Brighter gradient on hover */
}

.nav-button:active {
    transform: translateY(1px); /* Slight press-down effect */
    box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.2); /* Subtle shadow when active */
}


    /* Widgets Streamlit */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--text-color);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    /* Métriques */
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 2rem 0;
    }

    .metric-card {
        flex: 1;
        min-width: 200px;
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }

    .metric-title {
        font-size: 1.1rem;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--secondary-color);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.2s ease forwards;
    }

    /* Loading */
    .loading {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }

    .loading::after {
        content: '';
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid var(--secondary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive */
    @media (max-width: 768px) {
        .glass-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .metric-container {
            flex-direction: column;
        }

        .metric-card {
            width: 100%;
        }

        h1 {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

def load_model_and_transformers():
    """Charge les modèles et transformateurs nécessaires."""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {str(e)}")
        return None, None, None
    

def main():
    """Main application entry point."""
    import streamlit as st
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "main"
    
    # Custom CSS
    st.markdown("""
        <style>
        .fade-in {
            animation: fadeIn 1.5s;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
        }

        .card-image {
            width: 100%;
            height: 160px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            margin-bottom: 15px;
        }
        
        .powerbi-image {
            background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160"><rect width="200" height="160" fill="%232C3E50"/><rect x="40" y="100" width="20" height="40" fill="%233498DB"/><rect x="70" y="80" width="20" height="60" fill="%232980B9"/><rect x="100" y="60" width="20" height="80" fill="%233498DB"/><rect x="130" y="40" width="20" height="100" fill="%232980B9"/><path d="M40 90 Q100 30 150 20" stroke="%23E74C3C" stroke-width="3" fill="none"/><circle cx="40" cy="90" r="4" fill="%23E74C3C"/><circle cx="70" cy="70" r="4" fill="%23E74C3C"/><circle cx="100" cy="50" r="4" fill="%23E74C3C"/><circle cx="130" cy="30" r="4" fill="%23E74C3C"/><circle cx="150" cy="20" r="4" fill="%23E74C3C"/></svg>');
        }

        .ml-image {
            background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160"><rect width="200" height="160" fill="%232C3E50"/><circle cx="40" cy="40" r="10" fill="%239B59B6"/><circle cx="40" cy="80" r="10" fill="%239B59B6"/><circle cx="40" cy="120" r="10" fill="%239B59B6"/><circle cx="100" cy="60" r="10" fill="%238E44AD"/><circle cx="100" cy="100" r="10" fill="%238E44AD"/><circle cx="160" cy="80" r="10" fill="%239B59B6"/><path d="M50 40 L90 60" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 40 L90 100" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 80 L90 60" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 80 L90 100" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 120 L90 60" stroke="%23ECF0F1" stroke-width="2"/><path d="M50 120 L90 100" stroke="%23ECF0F1" stroke-width="2"/><path d="M110 60 L150 80" stroke="%23ECF0F1" stroke-width="2"/><path d="M110 100 L150 80" stroke="%23ECF0F1" stroke-width="2"/><path d="M170 80 L190 80" stroke="%23E74C3C" stroke-width="3"/><path d="M185 75 L190 80 L185 85" fill="none" stroke="%23E74C3C" stroke-width="3"/></svg>');
        }
        .visual-image {
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160"><rect width="200" height="160" fill="%232C3E50"/><rect x="30" y="90" width="20" height="50" fill="%233498DB"/><rect x="60" y="70" width="20" height="70" fill="%23E74C3C"/><rect x="90" y="50" width="20" height="90" fill="%239B59B6"/><rect x="120" y="30" width="20" height="110" fill="%23F1C40F"/><rect x="150" y="110" width="20" height="30" fill="%2349A44A"/><line x1="20" y1="140" x2="180" y2="140" stroke="%23FFFFFF" stroke-width="1"/><line x1="20" y1="10" x2="20" y2="140" stroke="%23FFFFFF" stroke-width="1"/><text x="10" y="145" fill="%23FFFFFF" font-size="8" font-family="Arial">0</text><text x="10" y="125" fill="%23FFFFFF" font-size="8" font-family="Arial">20</text><text x="10" y="85" fill="%23FFFFFF" font-size="8" font-family="Arial">60</text><text x="10" y="45" fill="%23FFFFFF" font-size="8" font-family="Arial">100</text><text x="10" y="15" fill="%23FFFFFF" font-size="8" font-family="Arial">140</text></svg>');

    }
        
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)
    

def main_page():
    """Page d'accueil avec design moderne."""
    st.markdown("""
        <style>
        /* Style de la bannière */
        .banner-container {
            position: relative;
            width: 100%;
            height: 300px;
            margin-bottom: 2rem;
            overflow: hidden;
            border-radius: 15px;
        }
        
        .banner-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }
        
        .banner-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(26, 28, 32, 0.85), rgba(44, 62, 80, 0.85));

            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            text-align: center;
            padding: 2rem;
        }
        
        .banner-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .banner-subtitle {
            font-size: 1.5rem;
            max-width: 800px;
            margin: 0 auto;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        /* Animation au survol */
        .banner-container:hover .banner-image {
            transform: scale(1.05);
        }
        
        /* Style des cartes amélioré */
        .glass-card {
            margin-top: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    # Bannière avec image et overlay
    st.markdown("""
        <div class="banner-container">
            <img src="https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80" 
                 class="banner-image" alt="Restaurant banner">
            <div class="banner-overlay">
                <h1 class="banner-title">Welcome to InsightPlate Analytics</h1>
                <p class="banner-subtitle">Discover powerful insights about restaurant performance and food safety in Los Angeles</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card fade-in">
        <h2>About This Platform :</h2>
        <h5>This platform uses advanced machine learning models to predict food quality, assess risk levels, 
            and evaluate employee compliance within the restaurant industry in Los Angeles. 
            <br>By leveraging real-time data 
            and predictive analytics, it aims to enhance food safety, optimize health inspections, and support restaurants 
            in maintaining high standards. 
        </h5>
    </div>
    """, unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

    with col1:
        st.markdown("""
                     <style>.card-title {
            color: inherit;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }</style>""", unsafe_allow_html=True)
        st.markdown("""
            <div class="glass-card">
                <div class="powerbi-image card-image"></div>
                <h3 class="card-title">Business Intelligence</h3>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Explore Analytics", key="powerbi_button", on_click=nav_to, args=('powerbi',)):
            st.session_state["page"] = "powerbi"

    with col2:
        st.markdown("""
                     <style>.card-title {
            color: inherit;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }</style>""", unsafe_allow_html=True)
        st.markdown("""
            <div class="glass-card">
                <div class="ml-image card-image"></div>
                <h3 class="card-title">Predictive Analytics</h3>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Access Predictions", key="ml_button", on_click=nav_to, args=('ml',)):
            st.session_state["page"] = "ml"
    with col3:
        st.markdown("""
                     <style>.card-title {
            color: inherit;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }</style>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="visual-image card-image"></div>
            <h3 class="card-title">Data Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Launch Insights", key="data_viz_button", on_click=nav_to, args=('viz',)):
            st.session_state["page"] = "viz"

def ml_page():
    """Page des modèles de machine learning."""
    st.markdown('<h1 class="fade-in">Predictive Models</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        
            <p>Choose from our suite of predictive models to gain valuable insights.</p>
        
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🍲 Food Quality Score Prediction", on_click=nav_to, args=('page_1',)):
            st.session_state["page"] = "page_1"

    with col2:
        if st.button("👥 Employee Compliance Rate", on_click=nav_to, args=('page_2',)):
            st.session_state["page"] = "page_2"

    with col3:
        if st.button("🏪 Restaurant Risk Level", on_click=nav_to, args=('page_3',)):
            st.session_state["page"] = "page_3"
            
    if st.button("← Return to Main Page", on_click=nav_to, args=('main',)):
        st.session_state["page"] = "main"

    st.image("res.png", use_container_width=True)
        
def page_1():
    """Page de prédiction du score de qualité alimentaire."""
    st.markdown('<h1 class="fade-in">Food Quality Score Predictor </h1>', unsafe_allow_html=True)
    st.markdown("""
        <style> * {
    color: white;
}  </style>
    """, unsafe_allow_html=True) 
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            model_xgb = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    try:
        df = pd.read_csv("dataset_adjusted_nulls.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return

    target_column = "Food_Quality_Score"
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataset.")
        return
    
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer_num.fit_transform(df[numerical_columns])

    if not categorical_columns.empty:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    X_reg = df.drop(columns=[target_column])
    y_reg = df[target_column]

    selector_reg = SelectKBest(score_func=f_regression, k=10)
    X_reg_selected = selector_reg.fit_transform(X_reg, y_reg)
    selected_features_reg = X_reg.columns[selector_reg.get_support()]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_selected, y_reg, test_size=0.2, random_state=42
    )

    y_train_xgb = model_xgb.predict(X_train_reg)
    y_pred_xgb = model_xgb.predict(X_test_reg)

    min_max_scaler = MinMaxScaler(feature_range=(0, 100))
    y_reg_scaled = min_max_scaler.fit_transform(y_reg.values.reshape(-1, 1))
    y_pred_scaled = min_max_scaler.transform(y_pred_xgb.reshape(-1, 1))
    y_test_scaled = min_max_scaler.transform(y_test_reg.values.reshape(-1, 1))

    mae_xgb = mean_absolute_error(y_test_scaled, y_pred_scaled)
    mse_xgb = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test_scaled, y_pred_scaled)

    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE (Mean Absolute Error)", f"{mae_xgb:.2f}")
    col2.metric("MSE (Mean Squared Error)", f"{mse_xgb:.2f}")
    col3.metric("RMSE (Root Mean Squared Error)", f"{rmse_xgb:.2f}")
    col4.metric("R² Score", f"{r2_xgb:.2f}")

    st.subheader("🔮 Predict Food Quality Score")
    st.write("Please enter the values of the selected variables to make a prediction:")
    user_input = {col: st.number_input(f"{col}", format="%.2f") for col in selected_features_reg}
    
    if st.button("Predict Food Quality Score") :
        user_input_array = np.array([list(user_input.values())]).reshape(1, -1)
        prediction = model_xgb.predict(user_input_array)

        user_prediction_scaled = min_max_scaler.transform(prediction.reshape(-1, 1))
        st.success(f"✅ The prediction of the Food Quality Score : {user_prediction_scaled[0][0]:.2f}")

    if st.button("Return to the machine learning page", on_click=nav_to, args=('ml',)):
        st.session_state["page"] = "ml" 


@st.cache_resource
def load_model_and_transformers():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders


def page_2():
    """Page de prédiction du taux de conformité des employés."""
    st.markdown('<h1 class="fade-in">Employee Compliance Rate Prediction</h1>', unsafe_allow_html=True)
 
    st.markdown(
    """
    <style>
    input {
        color: white !important;
        background-color: #333333 !important;
        border: 1px solid #555555 !important;
    }
    label {
        color: white !important;  /* Titre des inputs en blanc */
    }
     .stMetric > div {
        color: white !important;  /* Valeur prédite en blanc */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    model, scaler, label_encoders = load_model_and_transformers()
    certification_levels = label_encoders['Certification Level'].classes_
    with st.expander("About this predictor"):
        st.write("""This tool predicts employee compliance rates based on various factors including employee title, department, certification level, years of experience, inspection score, and training hours. The prediction uses a machine learning model based on historical data.""")
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Employee Information")
        # Employee Title (Text input)
        employee_title = st.text_input("Employee Title", placeholder="Enter the employee's title")
        
        # Department (Text input)
        department = st.text_input("Department", placeholder="Enter the department")
        
        # Certification Level (Select box with actual values from training data)
    


       #  Certification Level (Select box with actual values from training data)
        certification_level = st.selectbox(
       "Certification Level",
         certification_levels
)

    with col2:
        st.header("Performance Metrics")
        # Years of Experience (Slider: 0 to 39)
        years_of_experience = st.slider("Years of Experience", 0, 39, 5)
        
        # Avg Inspection Score (Slider: 0.0 to 10.0)
        avg_inspection_score = st.slider("Avg Inspection Score", 0.0, 10.0, 5.0)
        
        # Training Hours (Slider: 10 to 199)
        training_hours = st.slider("Training Hours", 10, 199, 50)
    

    if st.button("Predict Compliance Rate"):
        try:
            # Create input data frame
            input_data = pd.DataFrame({
                "Employee Title": [employee_title],
                "Years of Experience": [years_of_experience],
                "Avg Inspection Score": [avg_inspection_score],
                "Training Hours": [training_hours],
                "Certification Level": [certification_level],
                "Department": [department],
                "Compliance Rate": [0]  # Add dummy value for scaling
            })

            # Transform categorical variables using loaded label encoders
            for col, encoder in label_encoders.items():
                if col in input_data.columns:
                    try:
                        input_data[col] = encoder.transform(input_data[col])
                    except ValueError as e:
                        st.error(f"Error: Invalid value for {col}. Please check if the input matches the training data categories.")
                        st.write(f"Valid values for {col}:", encoder.classes_)
                        st.stop()

            # Scale all features using the loaded scaler
            scaled_features = scaler.transform(input_data)
            
            # Remove the Compliance Rate column for prediction
            scaled_features_no_target = np.delete(scaled_features, input_data.columns.get_loc("Compliance Rate"), axis=1)
            
            # Create DMatrix with proper feature names
            feature_names = [col for col in input_data.columns if col != "Compliance Rate"]
            dmatrix_input = xgb.DMatrix(scaled_features_no_target, feature_names=feature_names)
            
            # Make prediction
            scaled_pred = model.predict(dmatrix_input)
            
            # Create dummy array for inverse transform
            dummy_data = np.zeros_like(scaled_features)
            dummy_data[:, input_data.columns.get_loc("Compliance Rate")] = scaled_pred
            
            # Inverse transform prediction
            prediction = scaler.inverse_transform(dummy_data)[:, input_data.columns.get_loc("Compliance Rate")][0]
            
            # Display prediction with styling
            st.success("Prediction successful!")
            st.metric(
                label="Predicted Compliance Rate",
                value=f"{prediction:.1f}%",
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.checkbox("Show error details"):
                st.write("Error details:", e)
                st.write("Input data:", input_data)
            st.write("Please make sure all inputs are filled correctly and try again.")
            
    if st.button("Return to the machine learning page", on_click=nav_to, args=('ml',)):
        st.session_state["page"] = "ml" 


def extract_model(zip_file, output_folder):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

zip_file_path = 'good.zip'
output_folder = 'model/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
extract_model(zip_file_path, output_folder)

model_path = os.path.join(output_folder, 'good.pkl')
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Extraction may have failed.")

def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
def page_3():
    """Page de prédiction du niveau de risque des restaurants."""
    st.markdown('<h1 class="fade-in">Restaurant Risk Level Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
        <style> * {
    color: white;
}  </style>
    """, unsafe_allow_html=True) 
    # Chargement du modèle
    model = load_model()
    with st.expander("About this predictor"):
        st.write("""This tool predicts the **risk level** for restaurants in Los Angeles based on various inputs.
                 . The prediction uses a machine learning model based on historical data. """)

    
    class_labels = {
        0.0: "MODERATE RISK",
        1.0: "LOW RISK",
        2.0: "HIGH RISK"
    }

    st.subheader("🔮 Predict Risk Level")
    # Input fields for features
    st.subheader("""Please enter the values of the selected variables to make a prediction:""")
    feature1 = st.number_input('Violation Code')
    feature2 = st.number_input('Violation Status')
    feature3 = st.number_input('Grade')
    feature4 = st.number_input('Program Element (PE)')
    feature5 = st.number_input('Service Code')
    feature6 = st.number_input('Month')
    feature7 = st.number_input('Weekday')

    # Prepare input for the model
    user_input = pd.DataFrame(
        [[feature1, feature2, feature3, feature4, feature5, feature6, feature7]], 
        columns=['violation_code', 'violation_status', 'grade', 'program_element_pe', 
                'service_code','month', 'weekday']
    )
    
    # Predict button
    if st.button('Predict'):
        # Make a prediction
        prediction = model.predict(user_input)
        predicted_label = class_labels.get(float(prediction[0]), "Unknown")
        # Display the prediction result
        st.write(f"The prediction of Risk Level: **{predicted_label}**")
        
    if st.button("Return to the machine learning page", on_click=nav_to, args=('ml',)):
        st.session_state["page"] = "ml"  
def powerbi_page():
    """Page Power BI avec tableau de bord intégré."""
    st.markdown('<h1 class="fade-in">Analytics Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card">
            <div class="report-container">
                <iframe 
                    title="Restaurant Analytics Dashboard" 
                    src="https://app.powerbi.com/reportEmbed?reportId=d52a3c66-b1b0-4ac9-8bfe-e83383230933&autoAuth=true&embeddedDemo=true" 
                    frameborder="0" 
                    allowFullScreen="true" 
                    style="width: 100%; height: 600px;">
                </iframe>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("← Return to Main Page", on_click=nav_to, args=('main',)):
        st.session_state["page"] = "main"
    
def viz_page():
    st.markdown('<h1 class="fade-in">Data Insights</h1>', unsafe_allow_html=True)
    from streamlit.components.v1 import html
    st.markdown("""
    <div class="glass-card fade-in">
        <h5>
            Our platform leverages comprehensive restaurant inspection data to forecast food safety and compliance risks. 
            By analyzing key factors such as violation types, inspection outcomes, and employee performance, 
            we generate actionable insights that support the enhancement of restaurant safety standards in Los Angeles.
        </h5>
    </div>
   """, unsafe_allow_html=True)

    st.subheader("Datasets Overview")
    st.write("""
    This project uses three primary datasets related to restaurant and food market inspections in Los Angeles:
    - **Inspection Results Dataset**: Contains detailed records of food safety inspections in restaurants and food markets, including violation types and severity.
    - **Violation Types Dataset**: Provides a breakdown of common violations identified during inspections, helping to highlight frequent issues.
    - **Employee Performance Dataset**: Tracks the performance of restaurant staff during inspections, identifying potential correlations between employee behavior and food safety violations.
    """)
        
    st.subheader("Data Preprocessing and Feature Engineering")
    st.write("""
    The data was cleaned and preprocessed to ensure consistency and accuracy. Key transformations include:
    - Encoding categorical variables ( violation status, inspection outcomes...).
    - Handling missing values by imputing or removing records where appropriate.
    - Feature selection to identify the most relevant factors for predicting food safety risks.
    """)


    st.subheader("Decision Support for Food Safety")
    st.write("""
    The insights derived from the inspection data, combined with predictive analytics, provide a decision support tool for local authorities. 
    This platform allows authorities to identify at-risk establishments and allocate resources more effectively to improve food safety standards.
    """)

    m = folium.Map(location=[34.0522, -118.2437], zoom_start=13)
    marker = folium.Marker([34.0522, -118.2437], popup="Los Angeles")
    marker.add_to(m)
    # Display the map
    html(m._repr_html_(), height=470)
    
    if st.button("← Return to Main Page", on_click=nav_to, args=('main',)):
        st.session_state["page"] = "main"

def main():
    # Page routing
    pages = {
        'main': main_page,
        'ml': ml_page,
        'powerbi': powerbi_page,
        'viz': viz_page,
        'page_1': page_1,
        'page_2': page_2,
        'page_3': page_3
    }
    
    # Call the appropriate page function
    if st.session_state.page in pages:
        pages[st.session_state.page]()

if __name__ == "__main__":
    main()


# Gestion de la navigation
#if "page" not in st.session_state:
    #st.session_state["page"] = "main"

# Router
#page_mapping = {
    #"main": main_page,
    #"ml": ml_page,
   # "powerbi": powerbi_page,
   # "viz":viz_page,
    #"page_1": page_1,
   # "page_2": page_2,
  #  "page_3": page_3}

# Appel de la page appropriée
#current_page = st.session_state["page"]
#if current_page in page_mapping:
   # page_mapping[current_page]()
