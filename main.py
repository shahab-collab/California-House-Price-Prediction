import gradio as gr 
import pandas as pd  
import joblib

full_pipeline_model = joblib.load("full_pipeline_model.pkl")

DEFAULT_LONGITUDE = -119.57
DEFAULT_LATITUDE = 35.63

def predict_house_price(
    hosuing_median_age,
    median_income,
    population,
    total_rooms,
    total_bedrooms,
    households
):

    raw_data = pd.DataFrame([{
    "housing_median_age": housing_median_age,
    "median_income": median_income,
    "population": population,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "households": households,
    "longitude":DEFAULT_LONGITUDE,
    "latitude": DEFAULT_LATITUDE
    }])

    prediction = full_pipeline_model.predict(raw_data)

    return f"Predicted house value: ${prediction[0]:,.2f}"

demo = gr.Interface(
    fn = predict_house_price,
    inputs = [
        gr.Number(label = "hosuing median age"),
        gr.Number(label = "median income"),
        gr.Number(label = "population"),
        gr.Number(label = "total rooms"),
        gr.Number(label = "total bedrooms"),
        gr.Number(label = "households")
    ]
    outputs = "text",
    title = "California Housing Price Prediction",
    description = "Enter these 6 housing features to predict house price: "
)

import fastapi

app = fastapi.FastApi()
app = gr.mount_gradio_app(app, demo, path = "/")