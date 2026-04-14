import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
# metrics import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD ASSETS
@st.cache_resource
def load_assets():
    # Load LSTM and XGBoost models
    model_lstm = tf.keras.models.load_model('model_lstm.h5', compile=False) 
    suit_model = joblib.load('suitability_model.pkl') # Make sure you have this file!

    # Load Encoders and Scaler
    crop_encoder = joblib.load('crop_encoder.pkl')
    county_encoder = joblib.load('county_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Load Price Data
    price_df = pd.read_csv('cleaned_price_data.csv')
    price_df['County'] = price_df['County'].str.lower().str.strip()

    # Evaluation data
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')

    # return model_lstm, suit_model, crop_encoder, county_encoder, scaler, price_df
    return model_lstm, suit_model, crop_encoder, county_encoder, scaler, price_df, X_test, y_test

# Unpack all assets
# model_lstm, suit_model, crop_encoder, county_encoder, scaler, price_df = load_assets()
model_lstm, suit_model, crop_encoder, county_encoder, scaler, price_df, X_test, y_test = load_assets()

# 🌍 Environmental Data for All 47 Counties
county_env_data = {
    "baringo": {"temp": 27, "rain": 800, "ph": 6.5, "humidity": 55},
    "bomet": {"temp": 22, "rain": 1200, "ph": 6.3, "humidity": 70},
    "bungoma": {"temp": 25, "rain": 1400, "ph": 6.4, "humidity": 75},
    "busia": {"temp": 27, "rain": 1300, "ph": 6.2, "humidity": 78},
    "elgeyo-marakwet": {"temp": 23, "rain": 1000, "ph": 6.6, "humidity": 65},
    "embu": {"temp": 23, "rain": 1100, "ph": 6.1, "humidity": 68},
    "garissa": {"temp": 32, "rain": 400, "ph": 7.2, "humidity": 40},
    "homa-bay": {"temp": 26, "rain": 1200, "ph": 6.3, "humidity": 75},
    "isiolo": {"temp": 30, "rain": 500, "ph": 7.0, "humidity": 45},
    "kajiado": {"temp": 28, "rain": 600, "ph": 6.8, "humidity": 50},
    "kakamega": {"temp": 26, "rain": 1600, "ph": 6.4, "humidity": 82},
    "kericho": {"temp": 20, "rain": 1500, "ph": 5.9, "humidity": 80},
    "kiambu": {"temp": 21, "rain": 1200, "ph": 6.0, "humidity": 70},
    "kilifi": {"temp": 29, "rain": 900, "ph": 6.7, "humidity": 78},
    "kirinyaga": {"temp": 22, "rain": 1100, "ph": 6.1, "humidity": 68},
    "kisii": {"temp": 24, "rain": 1500, "ph": 6.3, "humidity": 85},
    "kisumu": {"temp": 28, "rain": 1300, "ph": 6.3, "humidity": 80},
    "kitui": {"temp": 29, "rain": 650, "ph": 6.9, "humidity": 50},
    "kwale": {"temp": 29, "rain": 1000, "ph": 6.6, "humidity": 80},
    "laikipia": {"temp": 24, "rain": 700, "ph": 6.8, "humidity": 55},
    "lamu": {"temp": 30, "rain": 900, "ph": 6.7, "humidity": 82},
    "machakos": {"temp": 27, "rain": 700, "ph": 6.7, "humidity": 55},
    "makueni": {"temp": 28, "rain": 650, "ph": 6.8, "humidity": 52},
    "mandera": {"temp": 34, "rain": 300, "ph": 7.3, "humidity": 35},
    "marsabit": {"temp": 29, "rain": 400, "ph": 7.0, "humidity": 40},
    "meru": {"temp": 22, "rain": 1400, "ph": 6.2, "humidity": 75},
    "migori": {"temp": 27, "rain": 1200, "ph": 6.4, "humidity": 78},
    "mombasa": {"temp": 30, "rain": 1100, "ph": 6.6, "humidity": 85},
    "muranga": {"temp": 21, "rain": 1300, "ph": 6.1, "humidity": 72},
    "nairobi": {"temp": 24, "rain": 1000, "ph": 6.5, "humidity": 65},
    "nakuru": {"temp": 23, "rain": 950, "ph": 6.8, "humidity": 60},
    "nandi": {"temp": 21, "rain": 1400, "ph": 6.2, "humidity": 75},
    "narok": {"temp": 24, "rain": 900, "ph": 6.7, "humidity": 60},
    "nyamira": {"temp": 23, "rain": 1500, "ph": 6.3, "humidity": 85},
    "nyandarua": {"temp": 18, "rain": 1100, "ph": 5.9, "humidity": 65},
    "nyeri": {"temp": 19, "rain": 1300, "ph": 5.8, "humidity": 78},
    "samburu": {"temp": 31, "rain": 500, "ph": 7.1, "humidity": 45},
    "siaya": {"temp": 27, "rain": 1200, "ph": 6.4, "humidity": 78},
    "taita-taveta": {"temp": 26, "rain": 800, "ph": 6.6, "humidity": 65},
    "tana-river": {"temp": 31, "rain": 600, "ph": 7.0, "humidity": 55},
    "tharaka-nithi": {"temp": 24, "rain": 900, "ph": 6.3, "humidity": 65},
    "trans-nzoia": {"temp": 22, "rain": 1200, "ph": 6.4, "humidity": 70},
    "turkana": {"temp": 35, "rain": 250, "ph": 7.5, "humidity": 30},
    "uasin-gishu": {"temp": 20, "rain": 900, "ph": 6.6, "humidity": 62},
    "vihiga": {"temp": 25, "rain": 1400, "ph": 6.4, "humidity": 80},
    "wajir": {"temp": 33, "rain": 300, "ph": 7.3, "humidity": 35},
    "west-pokot": {"temp": 28, "rain": 800, "ph": 6.7, "humidity": 55}
}

# 2. XGBOOST SUITABILITY LOGIC
def get_suitability_recommendations(county, temp_max, temp_min, rainfall, humidity, soil_ph):

    all_crops = crop_encoder.classes_
    county_enc = county_encoder.transform([county])[0]

    # 🔥 Allowed crops per county (REAL-WORLD FILTER)
    county_crop_map = {

    # 🌾 Rift Valley (High potential farming zone)
    "uasin-gishu": ["Wheat", "Dry Maize", "Green Maize", "White Irish Potatoes", "Cabbages", "Kales/Sukuma Wiki", "Carrots", "Spinach"],
    "nakuru": ["Wheat", "Dry Maize", "Green Maize", "White Irish Potatoes", "Cabbages", "Carrots", "Spinach", "Onions"],
    "narok": ["Wheat", "Dry Maize", "Green Maize", "Beans (Rosecoco)", "Cabbages"],
    "trans-nzoia": ["Wheat", "Dry Maize", "Green Maize", "Beans (Canadian wonder)", "Cabbages"],
    "elgeyo-marakwet": ["Finger Millet", "Green Grams", "Beans (Mwezi Moja)", "Sweet potatoes"],
    "nandi": ["Tea", "Dry Maize", "Beans (Rosecoco)", "Kales/Sukuma Wiki"],
    "kericho": ["Tea", "Dry Maize", "Beans (Rosecoco)", "Cabbages"],
    "bomet": ["Tea", "Dry Maize", "Beans (Rosecoco)", "Sweet potatoes"],
    "laikipia": ["Wheat", "Dry Maize", "Green Maize", "Beans (Canadian wonder)"],
    "west-pokot": ["Finger Millet", "Sorghum", "Green Grams", "Cowpeas"],
    "turkana": ["Sorghum", "Cowpeas", "Green Grams", "Water Melon"],
    "samburu": ["Sorghum", "Cowpeas", "Green Grams"],

    # 🌿 Western Kenya (High rainfall)
    "kakamega": ["Dry Maize", "Beans (Rosecoco)", "Banana (Cooking)", "Sweet potatoes", "Cassava Fresh", "Sugarcane"],
    "bungoma": ["Dry Maize", "Beans (Canadian wonder)", "Sweet potatoes", "Cassava Fresh"],
    "busia": ["Dry Maize", "Beans (Mwezi Moja)", "Cassava Fresh", "Sweet potatoes"],
    "vihiga": ["Dry Maize", "Beans (Rosecoco)", "Banana (Cooking)", "Sweet potatoes"],

    # 🌊 Nyanza
    "kisumu": ["Rice", "Sorghum", "Cassava Fresh", "Sweet potatoes"],
    "siaya": ["Sorghum", "Cassava Fresh", "Sweet potatoes", "Ground Nuts"],
    "homa-bay": ["Sorghum", "Cassava Fresh", "Sweet potatoes", "Ground Nuts"],
    "migori": ["Tobacco", "Sugarcane", "Cassava Fresh", "Sweet potatoes"],

    # 🌄 Central Highlands
    "kiambu": ["Tea", "Coffee", "Cabbages", "Carrots", "Spinach", "Kales/Sukuma Wiki"],
    "muranga": ["Tea", "Coffee", "Avocado", "Banana (Ripening)", "Macadamia Seed"],
    "nyeri": ["Tea", "Coffee", "Apples", "Plums", "Avocado"],
    "kirinyaga": ["Rice", "Banana (Cooking)", "Tomatoes", "French beans"],
    "nyandarua": ["White Irish Potatoes", "Carrots", "Cabbages", "Spinach"],

    # 🌱 Eastern
    "embu": ["Coffee", "Banana (Cooking)", "Avocado", "Maize"],
    "meru": ["Tea", "Coffee", "Banana (Cooking)", "Miraa", "Avocado"],
    "tharaka-nithi": ["Green Grams", "Millet", "Cowpeas"],
    "machakos": ["Green Grams", "Pigeon peas", "Cowpeas", "Sorghum"],
    "makueni": ["Green Grams", "Pigeon peas", "Mangoes", "Water Melon"],
    "kitui": ["Green Grams", "Pigeon peas", "Cowpeas", "Water Melon"],

    # 🏜️ ASAL North Eastern
    "garissa": ["Water Melon", "Cowpeas", "Green Grams"],
    "wajir": ["Cowpeas", "Green Grams"],
    "mandera": ["Cowpeas", "Sorghum"],

    # 🌴 Coastal Region
    "mombasa": ["Coconut", "Cassava Fresh", "Banana (Ripening)"],
    "kilifi": ["Coconut", "Cashewnuts (Korosho)", "Cassava Fresh", "Mangoes"],
    "kwale": ["Coconut", "Cashewnuts (Korosho)", "Cassava Fresh", "Pineapples"],
    "taita-taveta": ["Mangoes", "Banana (Ripening)", "Tomatoes"],
    "lamu": ["Coconut", "Cashewnuts (Korosho)", "Cassava Fresh"],
    "tana-river": ["Rice", "Water Melon", "Banana (Ripening)"],

    # 🌆 Nairobi (urban farming)
    "nairobi": ["Kales/Sukuma Wiki", "Spinach", "Tomatoes", "Cabbages"],

    # 🌾 Upper Eastern / Semi-arid
    "isiolo": ["Water Melon", "Cowpeas"],
    "marsabit": ["Sorghum", "Cowpeas"],
    "garissa": ["Water Melon", "Cowpeas"],

    # 🌊 Additional
    "kisii": ["Banana (Cooking)", "Tea", "Coffee", "Avocado"],
    "nyamira": ["Banana (Cooking)", "Tea", "Coffee"],

    # 🧩 Fill missing counties safely
    "kajiado": ["Tomatoes", "Onions", "Water Melon"],
    "narok": ["Wheat", "Dry Maize"],
    "bomet": ["Tea", "Maize"],
    "trans-nzoia": ["Wheat", "Maize"]
    }

    allowed_crops = county_crop_map.get(county, all_crops)

    recommendations = []

    # Compute Avg Temp (same as training)
    temp_avg = (temp_max + temp_min) / 2

    for crop_name in allowed_crops:  # 🔥 ONLY allowed crops

        try:
            # crop_enc = crop_le.transform([crop_name])[0]
            crop_enc = crop_encoder.transform([crop_name])[0]
        except:
            continue

        features = pd.DataFrame([[
            county_enc,
            crop_enc,
            temp_avg,
            rainfall,
            humidity,
            soil_ph
        ]], columns=[
            'County_Enc',
            'Crop_Enc',
            'Temp_Avg',
            'Rainfall_Annualized',
            'Humidity_Smooth',
            'Soil_pH_Smooth'
        ])

        # Predict
        is_suitable = suit_model.predict(features)[0]
        probability = suit_model.predict_proba(features)[0][1]

        # 🔥 STRONG FILTERS
        if is_suitable == 1 and probability >= 0.85:
            recommendations.append({
                "Crop": crop_name,
                "Match_Score": round(probability * 100, 2)
            })

    return sorted(recommendations, key=lambda x: x['Match_Score'], reverse=True)

# 3. HYBRID LOGIC
def test_hybrid_system(county_name, temp, rain, ph, humidity):
    search_county = county_name.lower().strip()

    # 1. Environment Suitability (XGBoost)
    suitable_crops = get_suitability_recommendations(
        county=search_county, 
        temp_max=temp, 
        temp_min=temp-5, 
        rainfall=rain, 
        humidity=humidity, 
        soil_ph=ph
    )

    if not suitable_crops:
        return []

    final_recommendations = []

    # 2. Predict Prices for survivors (LSTM)
    # We already have county_id from the encoder
    county_id = county_encoder.transform([search_county])[0]

    for item in suitable_crops:
        crop_name = item['Crop']
        suit_score = item['Match_Score'] / 100

        history_df = price_df[(price_df['Commodity'] == crop_name) & 
                              (price_df['County'] == search_county)].iloc[-30:]

        if len(history_df) >= 5:
            # history_vals = history_df['Wholesale'].values.reshape(1, 30, 1)
            prices = history_df['Wholesale'].values

            # Pad if less than 30
            if len(prices) < 30:
                prices = np.pad(prices, (30 - len(prices), 0), mode='edge')

            history_vals = prices.reshape(1, 30, 1)

            try:
                crop_id = crop_encoder.transform([crop_name])[0]
                meta = np.array([[crop_id, county_id]])

                pred_scaled = model_lstm.predict([history_vals, meta], verbose=0)

                # Inverse Scale (Dummy Array Fix)
                dummy = np.zeros((1, 2))
                dummy[0, 0] = pred_scaled[0, 0]
                pred_price = scaler.inverse_transform(dummy)[0, 0]

                # Get Market (latest)
                market = history_df.iloc[-1]['Market'] if not history_df.empty else "Unknown"
                # Simple planting logic (can improve later)
                if rain > 1000:
                    planting_time = "Start of rainy season"
                elif rain > 700:
                    planting_time = "Moderate rainfall period"
                else:
                    planting_time = "Irrigation recommended"

                final_recommendations.append({
                    "Crop": crop_name,
                    "Suitability": f"{item['Match_Score']}%",
                    "Forecasted_Price (KES)": round(pred_price, 2),
                    "Market": market,
                    "Best Planting Time": planting_time,
                    "Index": round(suit_score * pred_price, 2)
                })
            except:
                continue

    return sorted(final_recommendations, key=lambda x: x['Index'], reverse=True)

# 4. INTERFACE
st.set_page_config(page_title="Crop Recommender", page_icon="🌱", layout="wide")
st.title("Crop Recommendation System")
st.markdown("---")

with st.sidebar:
    st.header("📍 Select Location")
    c = st.selectbox("Select County", options=sorted(county_env_data.keys()))

if st.button("Get Crop Recommendations"):

    env = county_env_data.get(c)

    if env:
        st.info(
            f"Using environmental data for {c.upper()}:\n\n"
            f"🌡️ Temperature: {env['temp']} °C\n"
            f"🌧️ Rainfall: {env['rain']} mm\n"
            f"🧪 Soil pH: {env['ph']}\n"
            f"💧 Humidity: {env['humidity']}%"
        )

        with st.spinner("Analyzing biological and economic data..."):
            results = test_hybrid_system(
                c,
                temp=env["temp"],
                rain=env["rain"],
                ph=env["ph"],
                humidity=env["humidity"]
            )

        if results:
            df = pd.DataFrame(results)
            # st.success(f"Top Recommendation for {c}: **{results[0]['Crop']}**")

            top = results[0]

            st.success(
                f"""
                🌱 **Best Crop: {top['Crop']}**

                💰 Price: KES {top['Forecasted_Price (KES)']}  
                🏪 Market: {top['Market']}  
                📅 Planting Time: {top['Best Planting Time']}  
                """
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df, use_container_width=True)
            with col2:
                st.bar_chart(data=df, x="Crop", y="Index")
        else:
            st.error("No suitable crops found or missing price data for this county.")

    else:
        st.error("County data not found.")

# show metrics
st.markdown("---")
st.header("Model Performance Evaluation")

if st.checkbox("Show Model Performance"):

    # Predictions
    y_pred = suit_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Unsuitable", "Suitable"],
        yticklabels=["Unsuitable", "Suitable"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
