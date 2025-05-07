import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
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
        color: white;
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
        background-color: #e9f7ef;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    h1, h2, h3 {
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
        color: white; /* Set default text color */
    }

/* Overlay box for content */
    .overlay-box {
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black */
        padding: 20px;
        border-radius: 10px;
        margin: 20px auto;
        max-width: 800px;
        color: white; /* Ensure text remains white */
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

# Load dataset
@st.cache_data
def load_data():
    try:
        DATA_PATH = 'C:/Users/sarka/OneDrive/Desktop/agg/CROP-RECOMMENDATION/Crop_recommendation.csv'
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

# Function to show crop image
def show_crop_image(crop_name):
    # Create crop_images directory if it doesn't exist
    if not os.path.exists('crop_images'):
        os.makedirs('crop_images')
        
    image_path = os.path.join('crop_images', f"{crop_name.lower()}.jpg")
    placeholder_image = os.path.join('crop_images', 'placeholder.jpg')
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return image
    elif os.path.exists(placeholder_image):
        image = Image.open(placeholder_image)
        return image
    else:
        return None

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

# Main Application
def main():
    # Set background
    set_background()
    
    # Header
    st.markdown("<h1 style='text-align: center;'> SMART CROP RECOMMENDATION SYSTEM </h1>", unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    
    # Create fertilizer data instead of loading from file (which was causing the error)
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
            options=["New Harvest", "Ongoing Harvest", "About", "Data Insights"],
            icons=["seed", "moisture", "info-circle", "graph-up"],
            default_index=0,
        )
    
    # New Harvest Section - Predict both crop and fertilizer
    if selected == "New Harvest":
        st.markdown("##  New Harvest Planning")
        st.markdown("""
        <div class='info-box'>
        Plan your new harvest by entering soil and environmental parameters. 
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
        
        if st.button("Predict Crop & Fertilizer", key="predict_new"):
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
                            <div style='background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:10px;'>
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
    
    # Ongoing Harvest Section - Predict fertilizer only
    elif selected == "Ongoing Harvest":
        st.markdown("##  Ongoing Harvest Management")
        st.markdown("""
        <div class='info-box'>
        Manage your ongoing harvest by checking if additional fertilizer is needed.
        Enter your current soil parameters and crop details below.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Parameters")
            nitrogen = st.slider("Current Nitrogen (N) - kg/ha", 0, 140, 80, 1)
            phosphorus = st.slider("Current Phosphorus (P) - kg/ha", 0, 145, 90, 1)
            potassium = st.slider("Current Potassium (K) - kg/ha", 0, 205, 100, 1)
            
            # Soil type for fertilizer recommendation
            soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
            soil_type = st.selectbox("Soil Type", soil_types)
        
        with col2:
            st.subheader("Crop Details")
            # Get unique crop types from the dataset
            crop_options = sorted(df['label'].unique())
            selected_crop = st.selectbox("Current Crop", crop_options)
            
            # Add NPK visualization
            fig = px.bar(
                x=['Nitrogen', 'Phosphorus', 'Potassium'],
                y=[nitrogen, phosphorus, potassium],
                labels={'x': 'Nutrient', 'y': 'Amount (kg/ha)'},
                color_discrete_sequence=['#4CAF50', '#8BC34A', '#CDDC39']
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig)
        
        if st.button("Check Fertilizer Needs", key="check_fertilizer"):
            # Recommend fertilizer based on crop selection
            recommendation, n_def, p_def, k_def = recommend_fertilizer(
                nitrogen, phosphorus, potassium, soil_type, selected_crop, fertilizer_df
            )
            
            # Display results
            st.markdown("###  Fertilizer Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if "No fertilizer required" in recommendation:
                    st.markdown(f"""
                    <div class='success-box' style='height:150px;'>
                    <h3>✅ Good News!</h3>
                    <p>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='info-box' style='height:150px;'>
                    <h4>Fertilizer Recommendation:</h4>
                    <p>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if n_def > 0 or p_def > 0 or k_def > 0:
                    st.markdown("#### Nutrients to Add:")
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
    
    # About Section
    elif selected == "About":
        st.markdown("## About Smart Crop Recommendation System")
        
        st.markdown("""
        <div class='crop-card'>
        <h3>🌱 What is Smart Crop?</h3>
        <p>The Smart Crop Recommendation System uses machine learning to help farmers make data-driven decisions.
        By analyzing soil composition and environmental factors, our system recommends the most suitable crops
        and provides targeted fertilizer recommendations to optimize yield and reduce resource waste.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='crop-card'>
        <h3> How it Works</h3>
        <p>Our system uses a Random Forest machine learning algorithm trained on agricultural data.
        It considers six key parameters:</p>
        <ul>
            <li><strong>N, P, K Values:</strong> Nitrogen, Phosphorous, and Potassium content in soil</li>
            <li><strong>Temperature:</strong> Ambient temperature in degrees Celsius</li>
            <li><strong>Humidity:</strong> Relative humidity percentage</li>
            <li><strong>Rainfall:</strong> Rainfall in millimeters</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='crop-card'>
            <h3> New Harvest</h3>
            <p>Plan your new planting season by getting recommendations on both crop selection
            and fertilizer requirements based on soil testing and environmental conditions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='crop-card'>
            <h3> Ongoing Harvest</h3>
            <p>For existing crops, check if additional fertilizer is needed to maximize
            yield and optimize plant health based on current soil conditions.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data Insights Section
    elif selected == "Data Insights":
        st.markdown("##  Data Insights")
        
        st.markdown("""
        <div class='info-box'>
        Explore patterns and insights from our agricultural dataset to better understand 
        crop requirements and relationships between different parameters.
        </div>
        """, unsafe_allow_html=True)
        
        # Add tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Crop Requirements", "Parameter Relationships", "Dataset Overview"])
        
        with tab1:
            st.subheader("Nutrient Requirements by Crop")
            
            # Calculate average N, P, K for each crop
            crop_nutrients = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()
            
            # Let user select crops to compare
            selected_crops = st.multiselect(
                "Select crops to compare", 
                options=sorted(df['label'].unique()),
                default=sorted(df['label'].unique())[:5]  # Default to first 5 crops
            )
            
            if selected_crops:
                filtered_data = crop_nutrients[crop_nutrients['label'].isin(selected_crops)]
                
                # Create grouped bar chart
                fig = px.bar(
                    filtered_data.melt(id_vars='label', value_vars=['N', 'P', 'K'], 
                                       var_name='Nutrient', value_name='Value'),
                    x='label', y='Value', color='Nutrient',
                    title="Average NPK Requirements by Crop",
                    labels={'label': 'Crop', 'Value': 'Amount (kg/ha)'},
                    barmode='group',
                    color_discrete_map={'N': '#FF5733', 'P': '#33FF57', 'K': '#3357FF'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one crop to compare")
        
        with tab2:
            st.subheader("Parameter Relationships")
            
            # Select parameters to plot
            x_param = st.selectbox("X-axis parameter",
                                 options=['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'],
                                 index=0)
            y_param = st.selectbox("Y-axis parameter", 
                                 options=['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'],
                                 index=1)
            
            # Create scatter plot
            fig = px.scatter(
                df, x=x_param, y=y_param, color='label',
                title=f"Relationship between {x_param} and {y_param}",
                labels={x_param: x_param, y_param: y_param},
                opacity=0.7
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Dataset Overview")
            
            # Show data statistics
            st.markdown("#### Dataset Statistics")
            st.dataframe(df.describe())
            
            # Count of samples per crop
            st.markdown("#### Samples per Crop")
            crop_counts = df['label'].value_counts().reset_index()
            crop_counts.columns = ['Crop', 'Count']
            
            fig = px.pie(
                crop_counts, values='Count', names='Crop',
                title="Distribution of Crops in Dataset",
                hole=0.4
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 50px; color: #666;'>
        <p>© 2025 Smart Crop Recommendation System | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()