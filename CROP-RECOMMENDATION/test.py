import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to find file in directories
def find_file(filename, search_paths=None):
    if search_paths is None:
        # Default paths to search
        search_paths = [
            '.',  # Current directory
            './CROP-RECOMMENDATION',
            '../CROP-RECOMMENDATION',
            './AgriSens-master/CROP-RECOMMENDATION',
            '../AgriSens-master/CROP-RECOMMENDATION',
            './Sens-master/AgriSens-master/CROP-RECOMMENDATION'
        ]
    
    # Add the filename to each path and check if it exists
    for path in search_paths:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            print(f"Found {filename} at {file_path}")
            return file_path
    
    # If we get here, the file wasn't found
    print(f"Could not find {filename} in any of the search paths.")
    return None

# Load the dataset
def load_data(file_path=None):
    if file_path is None:
        file_path = find_file('Crop_recommendation.csv')
    
    if file_path is None:
        print("Please enter the full path to your Crop_recommendation.csv file:")
        user_path = input().strip()
        if os.path.exists(user_path):
            file_path = user_path
        else:
            print(f"Error: File '{user_path}' not found.")
            return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Prepare the data
def prepare_data(df):
    # Check for missing values and duplicates
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Split features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, le

# Train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=20, random_state=5)
    model.fit(X_train, y_train)
    return model

# Save the trained model
def save_model(model, filename='RandomForest.pkl'):
    # Ask user for directory to save the model
    print(f"Enter directory to save the model (press Enter to use current directory):")
    save_dir = input().strip()
    
    if save_dir and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            save_dir = ''
    
    model_path = os.path.join(save_dir, filename) if save_dir else filename
    
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Model saved as {model_path}")
        return model_path
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

# Load a trained model
def load_model(filename=None):
    if filename is None:
        filename = find_file('RandomForest.pkl')
    
    if filename is None:
        print("Please enter the full path to your model file:")
        user_path = input().strip()
        if os.path.exists(user_path):
            filename = user_path
        else:
            print(f"Error: Model file '{user_path}' not found.")
            return None
    
    try:
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Predict crop based on input parameters
def predict_crop(model, input_data, label_encoder):
    # Make prediction
    prediction = model.predict(input_data)
    # Decode the prediction
    crop_name = label_encoder.inverse_transform(prediction)
    return crop_name[0]

# Plot feature importance
def plot_feature_importance(model, feature_names):
    # Get feature importances
    importances = model.feature_importances_
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance for Crop Recommendation')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Interactive prediction function
def interactive_prediction(model, label_encoder):
    print("\n=== Crop Recommendation System ===")
    print("Please enter the following information:")
    
    try:
        n = float(input("Nitrogen (N) content in soil (0-140): "))
        p = float(input("Phosphorous (P) content in soil (5-145): "))
        k = float(input("Potassium (K) content in soil (5-205): "))
        temp = float(input("Temperature in Celsius (8-45): "))
        humidity = float(input("Humidity in percentage (14-100): "))
        ph = float(input("pH value of soil (3-10): "))
        rainfall = float(input("Rainfall in mm (20-300): "))
        
        # Validation (very basic)
        if not (0 <= n <= 140 and 5 <= p <= 145 and 5 <= k <= 205 and 
                8 <= temp <= 45 and 14 <= humidity <= 100 and 
                3 <= ph <= 10 and 20 <= rainfall <= 300):
            print("Warning: Some values are outside typical ranges. Results may be unreliable.")
        
        # Prepare input data
        input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        
        # Make prediction
        crop = predict_crop(model, input_data, label_encoder)
        
        print(f"\nRecommended crop: {crop}")
        return input_data, crop
        
    except ValueError:
        print("Error: Please enter valid numerical values.")
        return None, None

# Main function
def main():
    print("=== AgriSens Crop Recommendation System ===")
    print("This system will help you determine the best crop to plant based on soil and environmental conditions.")
    
    # Load or train model
    try_load = input("Do you want to load a pre-trained model? (y/n): ").lower()
    
    if try_load == 'y':
        model = load_model()
        if model is None:
            print("Training new model instead...")
            df = load_data()
            if df is not None:
                X_train, X_test, y_train, y_test, le = prepare_data(df)
                model = train_model(X_train, y_train)
                save_model(model)
            else:
                print("Cannot proceed without data. Exiting.")
                return
        else:
            # Need to load data anyway for the label encoder
            df = load_data()
            if df is not None:
                _, _, _, _, le = prepare_data(df)
            else:
                print("Cannot proceed without data for label encoding. Exiting.")
                return
    else:
        df = load_data()
        if df is not None:
            X_train, X_test, y_train, y_test, le = prepare_data(df)
            model = train_model(X_train, y_train)
            save_option = input("Do you want to save the trained model? (y/n): ").lower()
            if save_option == 'y':
                save_model(model)
        else:
            print("Cannot proceed without data. Exiting.")
            return
    
    # Show feature importance
    if input("Do you want to see feature importance? (y/n): ").lower() == 'y':
        plot_feature_importance(model, df.drop('label', axis=1).columns)
    
    # Interactive prediction
    while True:
        input_data, crop = interactive_prediction(model, le)
        if input_data is not None and crop is not None:
            # Ask if the user wants to see similar crops
            if input("Do you want to see probability scores for other crops? (y/n): ").lower() == 'y':
                probs = model.predict_proba(input_data)[0]
                classes = le.classes_
                crop_probs = [(classes[i], prob) for i, prob in enumerate(probs)]
                crop_probs.sort(key=lambda x: x[1], reverse=True)
                
                print("\nCrop probabilities:")
                for crop_name, prob in crop_probs[:5]:  # Show top 5
                    print(f"{crop_name}: {prob:.4f}")
        
        if input("\nDo you want to make another prediction? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()