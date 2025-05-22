import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
# from twilio.rest import Client

df = pd.read_csv("./dataset/4-soils-dataset-new.csv")

# Load the ML model
try:
    model = joblib.load('./SavedModels/new-4-soils-stacking.joblib')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Twilio Configuration
# twilio_account_sid = "ACd81bfc749c28496c26b3b03dbb62985f"
# twilio_auth_token = "14623ec185ca7a9187c3561702702dea"
# twilio_phone_number = "+16207098293"
# recipient_phone_number = "+918567098852"

# client = Client(twilio_account_sid, twilio_auth_token)

# Crop Recommendations
crop_recommendations = {
    'Alluvial Soil': 'Rice, Wheat, Bajra, Chickpea, Soybean, Cotton, Mustard, Groundnut, Sesame, Barley, Maize',
    'Desert Soil': 'Corns, Beans, Onions, Garlics, Tomatoes.',
    'Red Soil': 'Cotton, Wheat, Rice, Pulses, Millets, Tobacco, Oil seeds, Potatoes, Fruits.',
    'Loamy Soil': 'Tomatoes, Carrot, Peppers, Green Beans, Cucumbers, Strawberries, Sweet Corn, Spinach, Potatoes.'
}

# Define images for each soil type
soil_images = {
    "Alluvial Soil": "./images/alluvial-soil.jpg",
    "Desert Soil": "./images/desert-soil.jpg",
    "Red Soil": "./images/red-soil.jpg",
    "Loamy Soil": "./images/loamy-soil.jpg",
}

# Sidebar Menu
with st.sidebar:
    st.title("Soil Predictor Application")
    st.write("An innovative application that predicts soil types based on input features such as Nitrogen, Phosphorus, Potassium, Humidity, Temperature, Conductivity, and pH.")
    selected = st.selectbox('Select Option', ['About Us', 'Soil Prediction', 'Data Visualization'])

# About Us Page
if selected == 'About Us':
    st.subheader("Welcome to Soil Predictor")
    st.markdown("""
    This application predicts soil types based on real-time input parameters such as pH, nitrogen, potassium, and other chemical attributes. 
    It uses a powerful ensemble machine learning model to assist farmers, researchers, and agri-tech professionals in understanding soil properties for informed decision-making.
    """)
    col1, col2 = st.columns(2)
    with col1: st.image("./images/1.jpg", use_container_width=True)
    with col2: st.image("./images/2.jpg", use_container_width=True)

    st.markdown("### 🔍 Key Features")
    st.markdown("""
    - 🚀 Predicts soil types using ensemble ML models
    - 📊 Displays graphs and feature insights
    - ⚙️ Real-time and batch predictions supported
    - 📱 Easy-to-use Streamlit interface
    """)

    st.markdown("### 🛠️ How to Use")
    st.markdown("""
    1. Enter the required soil parameters in the sidebar/input form.
    2. Click **Predict Soil** to view the soil type.
    3. Explore the graphical summary of your input data.
    """)

if selected == 'Data Visualization':
    st.subheader("🌱 Soil Type Classification Dashboard")
    st.markdown("""
    Welcome to the interactive soil analysis web app. This dashboard provides insights into soil properties,
    their relationships, and how they influence the classification of soil types using machine learning models.
    """)

    
    st.subheader("🧾 Dataset Preview")
    st.dataframe(df.head(2907))

    st.subheader("📊 Soil Type Distribution")
    st.bar_chart(df["Label"].value_counts())

    st.subheader("🔍 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.drop('Label', axis=1).corr(), annot=True, cmap='YlGnBu')
    st.pyplot(fig)

    st.subheader("📌 Feature Distributions")
    feature = st.selectbox("Choose a feature to visualize", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True)
    st.pyplot(fig)

    st.subheader("🔗 Pairwise Feature Relationships")
    fig = sns.pairplot(df, hue='Label')
    st.pyplot(fig)

    st.subheader("📈 Scatter Plot: Select Any Two Features")
    x_axis = st.selectbox("X-Axis", df.columns[:-1])
    y_axis = st.selectbox("Y-Axis", df.columns[:-1], index=1)
    fig = px.scatter(df, x=x_axis, y=y_axis, color='Label', title="Feature Relationship by Soil Type")
    st.plotly_chart(fig)




# Soil Prediction Page
if selected == 'Soil Prediction':
    # Input Fields
    col1, col2, col3 = st.columns(3)
    with col1: Nitrogen = st.text_input('Nitrogen')
    with col2: Phosphorus = st.text_input('Phosphorus')
    with col3: Potassium = st.text_input('Potassium')
    with col1: Temperature = st.text_input('Temperature')
    with col2: Conductivity = st.text_input('Conductivity')
    with col3: pH = st.text_input('pH')
    with col1: Moisture = st.text_input('Soil Moisture')

    # Soil Prediction Function
    def predict_soil(input_features):
        prediction = model.predict([input_features])[0]
        soil_types = ['Alluvial Soil', 'Desert Soil', 'Loamy Soil', 'Red Soil']
        return soil_types[prediction] if 0 <= prediction < len(soil_types) else "Unknown Soil Type"

    # Submit Button
    if st.button("Predict Soil"):
        try:
            input_features = [
                float(Nitrogen), float(Phosphorus), float(Potassium),
                float(Temperature), float(Conductivity), float(pH), float(Moisture)
            ]
            
            soil = predict_soil(input_features)

            # Alert if moisture is too low
            if float(Moisture) <= 20:
                st.error("Warning: Soil moisture is too low. Please consider irrigation.")

            # Display Soil Prediction
            st.success(f"According to the entered values, the soil appears to be {soil}.")

            # Send SMS with Soil Recommendation
            # sms_message = f"Your soil is {soil}. Recommended crops: {crop_recommendations[soil]}"
            # client.messages.create(body=sms_message, from_=twilio_phone_number, to=recipient_phone_number)

            # Display Crop Recommendations
            st.info(f"Recommended Crops: {crop_recommendations.get(soil, 'No recommendations available')}")
            
            # Show Image Based on Soil Type
            if soil in soil_images:
                st.image(soil_images[soil], width=250, caption=soil)
        
        except ValueError:
            st.error("Invalid input. Please enter valid numeric values for all fields.")

    # Clear Button
    if st.button("Clear"):
        st.rerun()
