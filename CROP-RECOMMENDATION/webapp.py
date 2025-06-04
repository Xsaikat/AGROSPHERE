import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import base64

# Set page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .info-box {
        background-color: rgba(0, 128, 0, 0.1); /* Light transparent green */
        border-left: 5px solid #4CAF50;         /* Solid green border */
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    .success-box {
        background-color: #000000;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    h1 {
        color: white;
    }
    h2, h3 {
        color: #2e7d32;
    }
    .stSidebar .css-1d391kg {
        background-color: #f0f5f0;
    }
    .crop-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .sensor-data {
        background-color: #f3f9f3;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .thingspeak-section {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    .data-source-toggle {
        background-color: #f0f5f0;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Function to set background image
def set_background():
    bg_img = """
    <style>
    
    .stApp {
        background-image: url("https://images.pexels.com/photos/1382102/pexels-photo-1382102.jpeg?cs=srgb&dl=pexels-todd-trapani-488382-1382102.jpg&fm=jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: yellow; /* Set default text color */
    }

/* Overlay box for content */
    .overlay-box {
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black */
        padding: 20px;
        border-radius: 10px;
        margin: 20px auto;
        max-width: 800px;
        color:rgba(0, 0, 0, 0.5); /* Ensure text remains white */
    }

    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        background-color: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }
    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)

# ThingSpeak Constants
CHANNEL_ID = '2749013'  # ThingSpeak Channel ID
READ_API_KEY = '651TK5QM17DNP9FA'  # ThingSpeak Read API Key

# Function to fetch ThingSpeak data from the last 2 minutes
def fetch_thingspeak_data():
    try:
        # Calculate timestamp for 2 minutes ago
        two_mins_ago = datetime.utcnow() - timedelta(minutes=2)
        start_time = two_mins_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Construct the API URL with time parameter
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&start={start_time}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'feeds' in data and len(data['feeds']) > 0:
                # Get the latest feed
                latest_feed = data['feeds'][-1]
                
                # Extract sensor values (field1: temperature, field2: humidity, etc.)
                temperature = float(latest_feed.get('field1', 0))
                humidity = float(latest_feed.get('field2', 0))
                rainfall = float(latest_feed.get('field8', 0))
                soil_moisture = float(latest_feed.get('field4', 0))
                nitrogen = float(latest_feed.get('field5', 0))
                phosphorus = float(latest_feed.get('field6', 0))
                potassium = float(latest_feed.get('field7', 0))
                light = float(latest_feed.get('field3', 0))
                
                return {
                    'temperature': temperature,
                    'humidity': humidity,
                    'rainfall': rainfall,
                    'soil_moisture': soil_moisture,
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'light': light,
                    'timestamp': latest_feed.get('created_at')
                }
            else:
                return None
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching ThingSpeak data: {str(e)}")
        return None

# Load dataset
@st.cache_data
def load_data():
    try:
        DATA_PATH = 'CROP-RECOMMENDATION/Crop_recommendation.csv'
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # Create a sample dataset if file not found
        st.warning("Dataset file not found. Using sample data instead.")
        
        # Create sample data
        crops = ['rice', 'maize', 'wheat', 'mungbean', 'Tea', 'cotton', 'coffee']
        data = []
        
        for crop in crops:
            for _ in range(10):  # 10 samples per crop
                data.append({
                    'N': np.random.randint(0, 140),
                    'P': np.random.randint(0, 145),
                    'K': np.random.randint(0, 205),
                    'temperature': np.random.uniform(10.0, 40.0),
                    'humidity': np.random.uniform(30.0, 95.0),
                    'rainfall': np.random.uniform(50.0, 300.0),
                    'label': crop
                })
        
        return pd.DataFrame(data)

# Create fertilizer recommendation data
@st.cache_data
def create_fertilizer_data():
    # Create a simple fertilizer dataset
    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['rice', 'maize', 'wheat', 'mungbean', 'Tea', 'cotton', 'coffee', 
                 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'blackgram',
                 'coconut', 'papaya', 'orange', 'apple', 'muskmelon', 'watermelon',
                 'grapes', 'mango', 'banana', 'pomegranate', 'lentil', 'jute']
    
    data = []
    
    for soil in soil_types:
        for crop in crop_types:
            # Create varied fertilizer recommendations for each soil-crop combination
            data.append({
                'Soil_Type': soil,
                'Crop_Type': crop,
                'N': np.random.randint(80, 140),
                'P': np.random.randint(40, 100),
                'K': np.random.randint(20, 80),
                'Fertilizer_Name': np.random.choice(['Urea', 'DAP', 'NPK', 'Ammonium Sulfate', 'MOP'])
            })
    
    return pd.DataFrame(data)

# Train and save model
@st.cache_resource
def train_model():
    df = load_data()
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest model for crop recommendation
    RF = RandomForestClassifier(n_estimators=20, random_state=5)
    RF.fit(X_train, y_train)
    
    # Save model
    RF_pkl_filename = 'RF_crop.pkl'
    with open(RF_pkl_filename, 'wb') as file:
        pickle.dump(RF, file)
    
    # Create a dummy fertilizer model (in real app, replace with actual model training)
    fertilizer_RF = RandomForestClassifier(n_estimators=20, random_state=5)
    fertilizer_RF.fit(X_train, y_train)  # Using same target for demonstration
    
    # Save fertilizer model
    fertilizer_pkl_filename = 'RF_fertilizer.pkl'
    with open(fertilizer_pkl_filename, 'wb') as file:
        pickle.dump(fertilizer_RF, file)
    
    return RF, fertilizer_RF

# Load models
@st.cache_resource
def load_models():
    try:
        # Load crop recommendation model
        with open('RF_crop.pkl', 'rb') as file:
            crop_model = pickle.load(file)
            
        # Load fertilizer recommendation model
        with open('RF_fertilizer.pkl', 'rb') as file:
            fertilizer_model = pickle.load(file)
            
        return crop_model, fertilizer_model
    except (EOFError, pickle.UnpicklingError, FileNotFoundError):
        # If models don't exist, train them
        return train_model()

# Function to make crop predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, rainfall, model):
    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, rainfall]])
    prediction = model.predict(features)
    
    # Get prediction probabilities
    proba = model.predict_proba(features)[0]
    # Get top 3 predictions
    top_indices = np.argsort(proba)[::-1][:3]
    top_crops = [model.classes_[i] for i in top_indices]
    top_probas = [proba[i] for i in top_indices]
    
    return prediction[0], top_crops, top_probas

# Function to recommend fertilizer
def recommend_fertilizer(nitrogen, phosphorus, potassium, soil_type, crop_type, fertilizer_df):
    try:
        # Find the recommended N, P, K values for the given soil and crop
        recommended = fertilizer_df[(fertilizer_df['Soil_Type'] == soil_type) & 
                                  (fertilizer_df['Crop_Type'] == crop_type)]
        
        if recommended.empty:
            return "No specific fertilizer recommendation found for this soil and crop combination.", None, None, None
        
        rec_N = recommended.iloc[0]['N']
        rec_P = recommended.iloc[0]['P']
        rec_K = recommended.iloc[0]['K']
        
        # Calculate deficiency
        n_deficiency = rec_N - nitrogen
        p_deficiency = rec_P - phosphorus
        k_deficiency = rec_K - potassium
        
        fertilizer_name = recommended.iloc[0]['Fertilizer_Name']
        
        # Determine if fertilizer is needed
        if n_deficiency <= 0 and p_deficiency <= 0 and k_deficiency <= 0:
            return "No fertilizer required. Soil nutrients are sufficient.", 0, 0, 0
        
        # Create recommendation
        recommendation = f"Recommended Fertilizer: {fertilizer_name}"
        
        return recommendation, n_deficiency, p_deficiency, k_deficiency
    
    except KeyError as e:
        # If the expected columns don't exist, create a generic recommendation
        st.error(f"Error in fertilizer recommendation: {e}. Using generic recommendations instead.")
        
        # Create a simple recommendation based on current NPK values
        n_deficiency = max(0, 100 - nitrogen)  # Assume optimal N is 100
        p_deficiency = max(0, 50 - phosphorus)  # Assume optimal P is 50
        k_deficiency = max(0, 50 - potassium)   # Assume optimal K is 50
        
        if n_deficiency <= 0 and p_deficiency <= 0 and k_deficiency <= 0:
            return "No fertilizer required. Soil nutrients are sufficient.", 0, 0, 0
        
        # Determine primary deficiency
        if n_deficiency > p_deficiency and n_deficiency > k_deficiency:
            fertilizer = "Nitrogen-rich fertilizer (e.g., Urea)"
        elif p_deficiency > n_deficiency and p_deficiency > k_deficiency:
            fertilizer = "Phosphorus-rich fertilizer (e.g., DAP)"
        else:
            fertilizer = "Potassium-rich fertilizer (e.g., MOP)"
            
        return f"Recommended Fertilizer: {fertilizer}", n_deficiency, p_deficiency, k_deficiency

# Function to fetch and return an online image silently
def show_crop_image(crop_name):
    try:
        query = crop_name + " crop"
        url = f"https://source.unsplash.com/600x400/?{query.replace(' ', '%20')}"

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
    except:
        pass  # Silently ignore all errors
    
    return None  # Return nothing if failed

# Function to create a radar chart for soil properties
def create_radar_chart(n, p, k, temp, humidity, rainfall):
    categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']
    
    # Normalize values for better visualization
    max_values = [140, 145, 205, 50, 100, 300]  # Approximate max values
    values = [n/max_values[0], p/max_values[1], k/max_values[2], 
              temp/max_values[3], humidity/max_values[4], rainfall/max_values[5]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Soil Properties',
        line_color='green',
        fillcolor='rgba(76, 175, 80, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=20, b=20)
    )
    
    return fig

# Function to visualize NPK deficiency
def create_npk_chart(n_def, p_def, k_def):
    if n_def == 0 and p_def == 0 and k_def == 0:
        return None
        
    # Only include positive deficiencies
    labels = []
    values = []
    colors = []
    
    if n_def > 0:
        labels.append('Nitrogen (N)')
        values.append(n_def)
        colors.append('#FF5733')
    
    if p_def > 0:
        labels.append('Phosphorus (P)')
        values.append(p_def)
        colors.append('#33FF57')
    
    if k_def > 0:
        labels.append('Potassium (K)')
        values.append(k_def)
        colors.append('#3357FF')
    
    if not labels:  # If no deficiencies
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title_text="Nutrient Deficiency Distribution",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Function to display sensor data from ThingSpeak
def display_sensor_data(sensor_data):
    if sensor_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Soil Parameters")
            st.markdown(f"""
            <div class="sensor-data">
                <p><strong>Nitrogen (N):</strong> {sensor_data['nitrogen']:.2f} mg/L</p>
                <p><strong>Phosphorus (P):</strong> {sensor_data['phosphorus']:.2f} mg/L</p>
                <p><strong>Potassium (K):</strong> {sensor_data['potassium']:.2f} mg/L</p>
                <p><strong>Soil Moisture:</strong> {sensor_data['soil_moisture']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Environmental Parameters")
            st.markdown(f"""
            <div class="sensor-data">
                <p><strong>Temperature:</strong> {sensor_data['temperature']:.2f} °C</p>
                <p><strong>Humidity:</strong> {sensor_data['humidity']:.2f}%</p>
                <p><strong>Rainfall:</strong> {sensor_data['rainfall']:.2f} mm</p>
                <p><strong>Light:</strong> {sensor_data['light']:.2f} lux</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"<p><em>Last updated: {sensor_data['timestamp']}</em></p>", unsafe_allow_html=True)
        
        # Display radar chart for sensor data
        st.plotly_chart(create_radar_chart(
            sensor_data['nitrogen'], 
            sensor_data['phosphorus'], 
            sensor_data['potassium'],
            sensor_data['temperature'],
            sensor_data['humidity'],
            sensor_data['rainfall']
        ))
        
        return True
    else:
        st.error("No sensor data available from ThingSpeak. Please check your connection or try again later.")
        return False

# Main Application
def main():
    # Set background
    set_background()
    
    # Initialize session state variables
    if 'refresh_thingspeak' not in st.session_state:
        st.session_state.refresh_thingspeak = False
    
    st.markdown("""
    <h1 style="color: white; text-align: center; 
               text-shadow: -1px -1px 0 #000, 
                            1px -1px 0 #000, 
                            -1px 1px 0 #000, 
                            1px 1px 0 #000;">
        SMART CROP RECOMMENDATION SYSTEM
    </h1>
    """, unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    
    # Create fertilizer data instead of loading from file
    fertilizer_df = create_fertilizer_data()
    
    crop_model, fertilizer_model = load_models()
    
    # Create a navigation menu
    with st.sidebar:
        try:
            st.image("download.png", width=200)
        except:
            st.image("https://www.clipartmax.com/png/small/187-1876520_agrosphere-farm-and-agriculture-logo.png", width=200)
        
        st.markdown("##  Agrosphere")
        
        selected = option_menu(
            menu_title=None,
            options=["Manual Input", "ThingSpeak Data", "Ongoing Harvest", "About", "Data Insights"],
            icons=["pencil-square", "cloud-download", "moisture", "info-circle", "graph-up"],
            default_index=0,
        )
    
    # Manual Input Section - User enters parameters manually
    if selected == "Manual Input":
        st.markdown("##  Manual Input - Crop & Fertilizer Recommendation")
        st.markdown("""
        <div class='info-box'>
        Enter soil and environmental parameters manually. 
        We'll recommend the best crop for your conditions and suggest fertilizer requirements.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Parameters")
            nitrogen = st.slider("Nitrogen (N) - kg/ha", 0, 140, 50, 1)
            phosphorus = st.slider("Phosphorus (P) - kg/ha", 0, 145, 50, 1)
            potassium = st.slider("Potassium (K) - kg/ha", 0, 205, 50, 1)
            
            # Soil type for fertilizer recommendation
            soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
            soil_type = st.selectbox("Soil Type", soil_types)
            
        with col2:
            st.subheader("Environmental Parameters")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 0.1)
            rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0, 0.1)
            
            # Add a radar chart for inputs
            st.plotly_chart(create_radar_chart(nitrogen, phosphorus, potassium, temperature, humidity, rainfall))
        
        if st.button("Predict Crop & Fertilizer", key="predict_manual"):
            inputs = np.array([nitrogen, phosphorus, potassium, temperature, humidity, rainfall])
            
            if np.isnan(inputs).any():
                st.error("Please enter valid values before predicting.")
            else:
                # Predict crop
                prediction, top_crops, top_probas = predict_crop(
                    nitrogen, phosphorus, potassium, temperature, humidity, rainfall, crop_model
                )
                
                # Display crop recommendation
                st.markdown("###  Crop Recommendation Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class='success-box'>
                    <h3>✅ Recommended Crop:</h3>
                    <h2 style='color:#2e7d32; text-align:center;'>{prediction.upper()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show crop image
                    image = show_crop_image(prediction)
                    if image:
                        st.image(image, caption=f"Recommended Crop: {prediction}", use_column_width=True)
                    else:
                        st.info(f"No image available for {prediction}")
                
                with col2:
                    # Show alternate recommendations
                    st.markdown("#### Alternative Options:")
                    for i in range(min(3, len(top_crops))):
                        if top_crops[i] != prediction:  # Skip the main prediction
                            confidence = top_probas[i] * 100
                            st.markdown(f"""
                            <div style='background-color:#000000; padding:10px; border-radius:5px; margin-bottom:10px;'>
                            <b>{top_crops[i]}</b> - Confidence: {confidence:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Create a bar chart for top predictions
                    fig = px.bar(
                        x=[crop for crop in top_crops if crop != prediction], 
                        y=[proba * 100 for crop, proba in zip(top_crops, top_probas) if crop != prediction],
                        labels={'x': 'Crop', 'y': 'Confidence (%)'},
                        title="Alternative Crop Confidence",
                        color_discrete_sequence=['#4CAF50']
                    )
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig)
                
                # Recommend fertilizer based on crop prediction
                st.markdown("###  Fertilizer Recommendation")
                
                recommendation, n_def, p_def, k_def = recommend_fertilizer(
                    nitrogen, phosphorus, potassium, soil_type, prediction, fertilizer_df
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class='info-box' style='height:150px;'>
                    <h4>Fertilizer Recommendation:</h4>
                    <p>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if n_def > 0 or p_def > 0 or k_def > 0:
                        st.markdown("#### Nutrient Requirements:")
                        st.markdown(f"""
                        - Nitrogen (N): {'Sufficient' if n_def <= 0 else f'Add {n_def:.1f} kg/ha'}
                        - Phosphorus (P): {'Sufficient' if p_def <= 0 else f'Add {p_def:.1f} kg/ha'}
                        - Potassium (K): {'Sufficient' if k_def <= 0 else f'Add {k_def:.1f} kg/ha'}
                        """)
                
                with col2:
                    # Show NPK deficiency chart
                    npk_chart = create_npk_chart(n_def, p_def, k_def)
                    if npk_chart:
                        st.plotly_chart(npk_chart)
                    else:
                        st.success("✅ No nutrient deficiencies detected!")
    
    # ThingSpeak Data Section - Automatically fetch data from ThingSpeak
    elif selected == "ThingSpeak Data":
        st.markdown("##  ThingSpeak Data - Automatic Crop & Fertilizer Recommendation")
        st.markdown("""
        <div class='thingspeak-section'>
        <h4>🔄 Auto-Fetching Data from ThingSpeak</h4>
        <p>This section automatically fetches sensor data from ThingSpeak for the last 2 minutes and provides crop and fertilizer recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define callback for refresh button
        def set_refresh():
            st.session_state.refresh_thingspeak = True
        
        # Add a refresh button with callback
        st.button("Refresh Sensor Data", key="refresh_button", on_click=set_refresh)
        
        # Check if we should refresh data
        should_fetch = st.session_state.refresh_thingspeak
        
        # Display loading spinner
        with st.spinner("Fetching data from ThingSpeak..."):
            # Fetch data from ThingSpeak
            sensor_data = fetch_thingspeak_data() if should_fetch else None
            # Reset the refresh flag
            st.session_state.refresh_thingspeak = False
        
        # Display the sensor data
        if sensor_data:
            st.success("✅ Data successfully fetched from ThingSpeak!")
            
            # Display sensor data in a nice format
            if display_sensor_data(sensor_data):
                # Soil type is required for fertilizer recommendation
                soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
                soil_type = st.selectbox("Select Soil Type for Fertilizer Recommendation", soil_types)
                
                # Convert sensor data for prediction
                # Note: ThingSpeak data might need scaling/conversion to match model's expected range
                nitrogen = sensor_data['nitrogen']
                phosphorus = sensor_data['phosphorus']
                potassium = sensor_data['potassium']
                temperature = sensor_data['temperature']
                humidity = sensor_data['humidity']
                rainfall = sensor_data['rainfall']
                
                # Predict button for ThingSpeak data
                if st.button("Predict Using ThingSpeak Data", key="predict_thingspeak"):
                    # Predict crop
                    prediction, top_crops, top_probas = predict_crop(
                        nitrogen, phosphorus, potassium, temperature, humidity, rainfall, crop_model
                    )
                    
                    # Display crop recommendation
                    st.markdown("###  Crop Recommendation Results")
                    # Display crop recommendation from ThingSpeak data
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div class='success-box'>
                        <h3>✅ Recommended Crop:</h3>
                        <h2 style='color:#2e7d32; text-align:center;'>{prediction.upper()}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show crop image
                        image = show_crop_image(prediction)
                        if image:
                            st.image(image, caption=f"Recommended Crop: {prediction}", use_column_width=True)
                        else:
                            st.info(f"No image available for {prediction}")
                    
                    with col2:
                        # Show alternate recommendations
                        st.markdown("#### Alternative Options:")
                        for i in range(min(3, len(top_crops))):
                            if top_crops[i] != prediction:  # Skip the main prediction
                                confidence = top_probas[i] * 100
                                st.markdown(f"""
                                <div style='background-color:#000000; padding:10px; border-radius:5px; margin-bottom:10px;'>
                                <b>{top_crops[i]}</b> - Confidence: {confidence:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Create a bar chart for top predictions
                        fig = px.bar(
                            x=[crop for crop in top_crops if crop != prediction], 
                            y=[proba * 100 for crop, proba in zip(top_crops, top_probas) if crop != prediction],
                            labels={'x': 'Crop', 'y': 'Confidence (%)'},
                            title="Alternative Crop Confidence",
                            color_discrete_sequence=['#4CAF50']
                        )
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig)
                    
                    # Recommend fertilizer based on crop prediction
                    st.markdown("###  Fertilizer Recommendation")
                    
                    recommendation, n_def, p_def, k_def = recommend_fertilizer(
                        nitrogen, phosphorus, potassium, soil_type, prediction, fertilizer_df
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class='info-box' style='height:150px;'>
                        <h4>Fertilizer Recommendation:</h4>
                        <p>{recommendation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if n_def > 0 or p_def > 0 or k_def > 0:
                            st.markdown("#### Nutrient Requirements:")
                            st.markdown(f"""
                            - Nitrogen (N): {'Sufficient' if n_def <= 0 else f'Add {n_def:.1f} kg/ha'}
                            - Phosphorus (P): {'Sufficient' if p_def <= 0 else f'Add {p_def:.1f} kg/ha'}
                            - Potassium (K): {'Sufficient' if k_def <= 0 else f'Add {k_def:.1f} kg/ha'}
                            """)
                    
                    with col2:
                        # Show NPK deficiency chart
                        npk_chart = create_npk_chart(n_def, p_def, k_def)
                        if npk_chart:
                            st.plotly_chart(npk_chart)
                        else:
                            st.success("✅ No nutrient deficiencies detected!")
        else:
            st.warning("No data available. Please click the 'Refresh Sensor Data' button to fetch the latest data.")
            
    # Ongoing Harvest Section - Hypothetical monitoring of current crops
    elif selected == "Ongoing Harvest":
        st.markdown("##  Ongoing Harvest Monitoring")
        st.markdown("""
        <div class='info-box'>
        Monitor your ongoing harvests and track their progress.
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data for ongoing harvests
        ongoing_harvests = [
            {"crop": "rice", "start_date": "2025-02-15", "expected_harvest": "2025-06-20", "area": 2.5, "health": 85},
            {"crop": "wheat", "start_date": "2025-03-10", "expected_harvest": "2025-07-05", "area": 3.0, "health": 92},
            {"crop": "maize", "start_date": "2025-04-01", "expected_harvest": "2025-08-15", "area": 1.8, "health": 78},
        ]
        
        # Display ongoing harvests
        for i, harvest in enumerate(ongoing_harvests):
            with st.expander(f"{harvest['crop'].capitalize()} - {harvest['area']} hectares", expanded=(i==0)):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Show crop image
                    image = show_crop_image(harvest['crop'])
                    if image:
                        st.image(image, caption=f"{harvest['crop'].capitalize()}", use_column_width=True)
                    else:
                        st.info(f"No image available for {harvest['crop']}")
                
                with col2:
                    # Calculate days until harvest
                    start_date = datetime.strptime(harvest['start_date'], "%Y-%m-%d")
                    expected_harvest = datetime.strptime(harvest['expected_harvest'], "%Y-%m-%d")
                    days_until_harvest = (expected_harvest - datetime.now()).days
                    
                    # Calculate progress percentage
                    total_days = (expected_harvest - start_date).days
                    days_passed = (datetime.now() - start_date).days
                    progress = min(100, max(0, (days_passed / total_days) * 100))
                    
                    st.markdown(f"**Start Date:** {harvest['start_date']}")
                    st.markdown(f"**Expected Harvest:** {harvest['expected_harvest']} ({days_until_harvest} days remaining)")
                    st.markdown(f"**Health Index:** {harvest['health']}%")
                    
                    # Progress bar
                    st.progress(int(progress))
                    st.caption(f"Growth Progress: {progress:.1f}%")
                    
                    # Add action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.button("View Details", key=f"details_{i}")
                    with col2:
                        st.button("Update Status", key=f"update_{i}")
                    with col3:
                        st.button("Weather Forecast", key=f"weather_{i}")
    
    # About Section - Information about the application
    elif selected == "About":
        st.markdown("##  About Smart Crop Recommendation System")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                st.image("logo.jpg", width=300)
            except:
                st.image("https://www.clipartmax.com/png/small/187-1876520_agrosphere-farm-and-agriculture-logo.png", width=300)
        
        with col2:
            st.markdown("""
            <div class='info-box'>
            <h3>Smart Agriculture for Sustainable Farming</h3>
            <p>The Smart Crop Recommendation System is designed to help farmers make informed decisions about crop selection and fertilizer usage based on soil and environmental conditions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Data Collection**: Soil and environmental data are collected either manually or through IoT sensors connected to ThingSpeak.
        2. **Data Processing**: The collected data is processed and analyzed by our machine learning models.
        3. **Recommendation**: Based on the analysis, the system recommends the most suitable crop and required fertilizers.
        4. **Monitoring**: Continuous monitoring of ongoing harvests helps detect issues early and maximize yield.
        """)
        
        st.markdown("### Benefits")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Increased Productivity**: Choose the right crop for the right conditions
            - **Cost Savings**: Use fertilizers only when and where needed
            - **Environmental Impact**: Reduce excessive fertilizer use
            """)
        
        with col2:
            st.markdown("""
            - **Time Savings**: Quick and accurate recommendations
            - **Data-Driven Decisions**: Base farming decisions on scientific analysis
            - **Sustainable Farming**: Promote eco-friendly agriculture practices
            """)
        
        st.markdown("### Technical Details")
        st.markdown("""
        The system uses a Random Forest Classifier trained on a comprehensive dataset of soil parameters, environmental conditions, and crop suitability. 
        The model has been optimized to provide accurate recommendations for various agricultural scenarios.
        
        **IoT Integration**: The system integrates with ThingSpeak for real-time data collection from soil sensors, weather stations, and other agricultural IoT devices.
        """)
    
    # Data Insights Section - Statistical analysis of the dataset
    elif selected == "Data Insights":
        st.markdown("##  Data Insights & Analytics")
        st.markdown("""
        <div class='info-box'>
        Explore statistical insights and patterns from our agricultural dataset.
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Crop Distribution", "Parameter Analysis", "Correlation Study"])
        
        with tab1:
            st.subheader("Crop Distribution in Dataset")
            
            # Count of crops in dataset
            crop_counts = df['label'].value_counts().reset_index()
            crop_counts.columns = ['Crop', 'Count']
            
            # Bar chart of crop distribution
            fig = px.bar(
                crop_counts, 
                x='Crop', 
                y='Count',
                title="Number of Samples per Crop",
                color='Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig)
            
            # Pie chart
            fig2 = px.pie(
                crop_counts, 
                values='Count', 
                names='Crop',
                title="Crop Distribution in Dataset",
                hole=0.4
            )
            st.plotly_chart(fig2)
        
        with tab2:
            st.subheader("Parameter Analysis by Crop")
            
            # Select parameter to analyze
            parameter = st.selectbox(
                "Select Parameter to Analyze",
                ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']
            )
            
            # Box plot of selected parameter by crop
            fig = px.box(
                df, 
                x='label', 
                y=parameter,
                title=f"{parameter.capitalize()} Distribution by Crop",
                color='label'
            )
            st.plotly_chart(fig)
            
            # Calculate statistics
            stats = df.groupby('label')[parameter].agg(['mean', 'median', 'min', 'max']).reset_index()
            stats.columns = ['Crop', 'Mean', 'Median', 'Minimum', 'Maximum']
            
            # Format to 2 decimal places
            for col in ['Mean', 'Median', 'Minimum', 'Maximum']:
                stats[col] = stats[col].round(2)
            
            st.markdown("#### Statistical Summary")
            st.dataframe(stats, width=800)
        
        with tab3:
            st.subheader("Correlation Between Parameters")
            
            # Select crop to analyze
            crop_options = ['All Crops'] + sorted(df['label'].unique().tolist())
            selected_crop = st.selectbox("Select Crop for Correlation Analysis", crop_options)
            
            # Filter data if specific crop selected
            if selected_crop != 'All Crops':
                filtered_df = df[df['label'] == selected_crop]
                title_suffix = f" for {selected_crop}"
            else:
                filtered_df = df
                title_suffix = " for All Crops"
            
            # Create correlation matrix
            corr_data = filtered_df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']].corr()
            
            # Plot heatmap
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Parameter Correlation Matrix" + title_suffix
            )
            st.plotly_chart(fig)
            
            # Scatter plot matrix for selected parameters
            selected_params = st.multiselect(
                "Select Parameters for Scatter Plot Matrix", 
                ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'],
                default=['N', 'P', 'K']
            )
            
            if len(selected_params) >= 2:
                fig = px.scatter_matrix(
                    filtered_df,
                    dimensions=selected_params,
                    color="label" if selected_crop == 'All Crops' else None,
                    title="Scatter Plot Matrix" + title_suffix
                )
                st.plotly_chart(fig)
            else:
                st.info("Please select at least 2 parameters for the scatter plot matrix")

if __name__ == "__main__":
    main()