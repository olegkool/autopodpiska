import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


with open('model/catboost_model.pkl', 'rb') as file:
    model: object = dill.load(file)


app = FastAPI()


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    result: int


@app.get('/status')
def status() -> object:
    return "I'm OK."


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
async def predict(form: Form):
    form_dict = form.dict()
    df = pd.DataFrame([form_dict])
    df.index = df['session_id']
    session_id = df['session_id']
    df = df.drop(columns=['session_id'])
    y = model['model'].predict(df)
    return {
        'session_id': f"{session_id[0]}",
        'result': y
    }

# Comands in Terminal:
# Conda activate
# Conda env list
# source activate sber_final
# pip install uvicorn (optional)
# uvicorn main:app --reload
# see on http://127.0.0.1:8000/docs#/