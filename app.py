import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Agriculture Assistant",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: white;
}

.main-title {
    text-align: center;
    color: #00BFFF;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 18px;
}

.result {
    text-align: center;
    color: #00BFFF;
    font-size: 30px;
    font-weight: bold;
}

div.stButton > button {
    width: 100%;
    background-color: #00BFFF;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px;
}

div.stButton > button:hover {
    background-color: #009ACD;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🌱 AI Agriculture Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart AI Solutions for Modern Agriculture'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌱 Agriculture Assistant")

option = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Crop Recommendation",
        "Fertilizer Recommendation",
        "Crop Yield Prediction",
        "Disease Prediction"
    ]
)


# =========================================================
# HOME
# =========================================================

if option == "Home":

    st.subheader("Welcome! 👋")

    st.write(
        "AI Agriculture Assistant is a machine learning based "
        "application designed to help farmers make better "
        "agricultural decisions."
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.info("🌾 Crop Recommendation")

        st.info("🧪 Fertilizer Recommendation")

    with col2:

        st.info("📈 Crop Yield Prediction")

        st.info("🍃 Plant Disease Prediction")


# =========================================================
# CROP RECOMMENDATION
# =========================================================

elif option == "Crop Recommendation":

    st.header("🌾 Crop Recommendation")

    st.write(
        "Enter soil and environmental values to get "
        "the recommended crop."
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    try:

        crop_model = joblib.load(
            os.path.join(
                "models",
                "best_crop_model.pkl"
            )
        )

        crop_encoder = joblib.load(
            os.path.join(
                "encoders",
                "crop_encoder.pkl"
            )
        )

        crop_loaded = True

    except Exception as e:

        st.error(
            f"Crop model loading error: {e}"
        )

        crop_loaded = False


    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=90.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=42.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=43.0
        )

        temperature = st.number_input(
            "Temperature",
            value=20.87
        )

    with col2:

        humidity = st.number_input(
            "Humidity",
            value=82.00
        )

        ph = st.number_input(
            "pH",
            value=6.50
        )

        rainfall = st.number_input(
            "Rainfall",
            value=202.93
        )


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button("🌱 Recommend Crop"):

        if crop_loaded:

            try:

                sample = pd.DataFrame({
                    "N": [nitrogen],
                    "P": [phosphorus],
                    "K": [potassium],
                    "temperature": [temperature],
                    "humidity": [humidity],
                    "ph": [ph],
                    "rainfall": [rainfall]
                })

                prediction = crop_model.predict(
                    sample
                )

                crop = crop_encoder.inverse_transform(
                    prediction
                )

                st.markdown("---")

                st.markdown(
                    '<div class="result">'
                    '🌱 Recommended Crop'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result">'
                    f'{crop[0]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =========================================================
# FERTILIZER RECOMMENDATION
# =========================================================

elif option == "Fertilizer Recommendation":

    st.header("🧪 Fertilizer Recommendation")

    st.write(
        "Enter soil, crop and nutrient values to get "
        "the recommended fertilizer."
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    try:

        fertilizer_model = joblib.load(
            os.path.join(
                "models",
                "fertilizer_model.pkl"
            )
        )

        soil_encoder = joblib.load(
            os.path.join(
                "encoders",
                "soil_encoder.pkl"
            )
        )

        fertilizer_crop_encoder = joblib.load(
            os.path.join(
                "encoders",
                "crop_encoder.pkl"
            )
        )

        fertilizer_encoder = joblib.load(
            os.path.join(
                "encoders",
                "fertilizer_encoder.pkl"
            )
        )

        scaler = joblib.load(
            os.path.join(
                "encoders",
                "scaler.pkl"
            )
        )

        fertilizer_loaded = True

    except Exception as e:

        st.error(
            f"Fertilizer model loading error: {e}"
        )

        fertilizer_loaded = False


    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    if fertilizer_loaded:

        col1, col2 = st.columns(2)

        with col1:

            temperature = st.number_input(
                "Temparature",
                value=26.0
            )

            humidity = st.number_input(
                "Humidity",
                value=60.0
            )

            moisture = st.number_input(
                "Moisture",
                value=40.0
            )

            nitrogen = st.number_input(
                "Nitrogen",
                min_value=0.0,
                value=20.0
            )

        with col2:

            potassium = st.number_input(
                "Potassium",
                min_value=0.0,
                value=10.0
            )

            phosphorous = st.number_input(
                "Phosphorous",
                min_value=0.0,
                value=15.0
            )

            soil = st.selectbox(
                "Soil Type",
                list(soil_encoder.classes_)
            )

            fertilizer_crop = st.selectbox(
                "Crop Type",
                list(
                    fertilizer_crop_encoder.classes_
                )
            )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        if st.button("🌱 Recommend Fertilizer"):

            try:

                soil_encoded = (
                    soil_encoder
                    .transform([soil])[0]
                )

                crop_encoded = (
                    fertilizer_crop_encoder
                    .transform(
                        [fertilizer_crop]
                    )[0]
                )

                input_data = pd.DataFrame(
                    [[
                        temperature,
                        humidity,
                        moisture,
                        soil_encoded,
                        crop_encoded,
                        nitrogen,
                        potassium,
                        phosphorous
                    ]],
                    columns=[
                        "Temparature",
                        "Humidity",
                        "Moisture",
                        "Soil Type",
                        "Crop Type",
                        "Nitrogen",
                        "Potassium",
                        "Phosphorous"
                    ]
                )

                input_scaled = scaler.transform(
                    input_data
                )

                prediction = (
                    fertilizer_model
                    .predict(input_scaled)
                )

                fertilizer = (
                    fertilizer_encoder
                    .inverse_transform(
                        prediction
                    )
                )

                st.markdown("---")

                st.markdown(
                    '<div class="result">'
                    '🌱 Recommended Fertilizer'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result">'
                    f'{fertilizer[0]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =========================================================
# CROP YIELD PREDICTION
# =========================================================

elif option == "Crop Yield Prediction":

    st.header("📈 Crop Yield Prediction")

    st.write(
        "Enter agricultural and environmental information "
        "to predict crop yield."
    )

    # -----------------------------------------------------
    # LOAD MODEL & ENCODERS
    # -----------------------------------------------------

    try:

        yield_model = joblib.load(
            os.path.join(
                "models",
                "crop_yield_model.pkl"
            )
        )

        area_encoder = joblib.load(
            os.path.join(
                "encoders",
                "area_encoder.pkl"
            )
        )

        item_encoder = joblib.load(
            os.path.join(
                "encoders",
                "item_encoder.pkl"
            )
        )

        yield_loaded = True

    except Exception as e:

        st.error(
            f"Yield model loading error: {e}"
        )

        yield_loaded = False


    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    if yield_loaded:

        col1, col2 = st.columns(2)

        with col1:

            area = st.selectbox(
                "Area",
                list(
                    area_encoder.classes_
                )
            )

            item = st.selectbox(
                "Crop Name",
                list(
                    item_encoder.classes_
                )
            )

            year = st.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=2025,
                step=1
            )

        with col2:

            rainfall = st.number_input(
                "Average Rainfall (mm/year)",
                min_value=0.0,
                value=1000.0
            )

            pesticides = st.number_input(
                "Pesticides (tonnes)",
                min_value=0.0,
                value=100.0
            )

            temperature = st.number_input(
                "Average Temperature",
                value=25.0
            )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        if st.button("📈 Predict Crop Yield"):

            try:

                area_encoded = (
                    area_encoder
                    .transform([area])[0]
                )

                item_encoded = (
                    item_encoder
                    .transform([item])[0]
                )

                prediction = (
                    yield_model.predict(
                        [[
                            area_encoded,
                            item_encoded,
                            year,
                            rainfall,
                            pesticides,
                            temperature
                        ]]
                    )
                )

                predicted_yield = prediction[0]

                st.markdown("---")

                st.markdown(
                    '<div class="result">'
                    '📈 Predicted Crop Yield'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result">'
                    f'{predicted_yield:.2f} hg/ha'
                    f'</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =========================================================
# DISEASE PREDICTION
# =========================================================

elif option == "Disease Prediction":

    st.header("🍃 Plant Disease Prediction")

    st.write(
        "Upload a plant leaf image to predict its disease."
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    try:

        disease_model = tf.keras.models.load_model(
            os.path.join(
                "models",
                "disease_model.keras"
            )
        )

        disease_loaded = True

    except Exception as e:

        st.error(
            f"Disease model loading error: {e}"
        )

        disease_loaded = False


    # -----------------------------------------------------
    # CLASS NAMES
    # -----------------------------------------------------

    class_names = [

        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",

        "Blueberry___healthy",

        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",

        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",

        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",

        "Orange___Haunglongbing_(Citrus_greening)",

        "Peach___Bacterial_spot",
        "Peach___healthy",

        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",

        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",

        "Raspberry___healthy",
        "Soybean___healthy",

        "Squash___Powdery_mildew",

        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",

        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]


    # -----------------------------------------------------
    # UPLOAD IMAGE
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Leaf Image",
            width=300
        )


        if st.button("🍃 Predict Disease"):

            if disease_loaded:

                try:

                    # -----------------------------------------
                    # LOAD IMAGE DIRECTLY
                    # -----------------------------------------

                    img = tf.keras.utils.load_img(
                        uploaded_file,
                        target_size=(128, 128)
                    )

                    img_array = (
                        tf.keras.utils
                        .img_to_array(img)
                    )

                    img_array = (
                        img_array / 255.0
                    )

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )


                    # -----------------------------------------
                    # PREDICTION
                    # -----------------------------------------

                    prediction = (
                        disease_model.predict(
                            img_array,
                            verbose=0
                        )
                    )

                    predicted_class = (
                        np.argmax(prediction)
                    )

                    confidence = (
                        np.max(prediction)
                        * 100
                    )


                    # -----------------------------------------
                    # RESULT
                    # -----------------------------------------

                    st.markdown("---")

                    st.markdown(
                        '<div class="result">'
                        '🍃 Predicted Disease'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="result">'
                        f'{class_names[predicted_class]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        f"Confidence: "
                        f"{confidence:.2f}%"
                    )


                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )