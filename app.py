from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from joblib import load
import json

app = FastAPI()

# Пример данных и кодировок
data = {
    'Country': ['USA', 'Canada'],
    'State': ['California', 'Ontario'],
    'City': ['Los Angeles', 'Toronto'],
    'Region': ['West', 'East'],
    'Segment': ['Consumer', 'Corporate'],
    'Ship Mode': ['Standard Class', 'Second Class'],
    'Category': ['Furniture', 'Technology'],
    'Sub-Category': ['Chairs', 'Phones'],
    'Discount': [0.1, 0.2],
    'Sales': [2000, 1500],
    'Profit': [300, 200],
    'Quantity': [10, 5],
    'Feedback?': [True, False],
    'Price_per_unit': [200, 300],
    'Profit_margin': [0.15, 0.13],
    'Discount_flag': [1, 1],
    'Has_Feedback': [1, 0]
}
df = pd.DataFrame(data)

categorical_features = ['Country', 'State', 'City', 'Region', 'Segment', 'Ship Mode', 'Category', 'Sub-Category']

encoding_dict = {}
for col in categorical_features:
    unique_vals = df[col].unique().tolist()
    encoding_dict[col] = {val: idx for idx, val in enumerate(unique_vals)}

# Загрузка модели
model = load('best_model.pkl')

@app.get("/", response_class=HTMLResponse)
def read_root():
    example_post = {
        "features": [
            2000,     # Sales
            10,       # Quantity
            0.1,      # Discount
            200,      # Price_per_unit
            0.15,     # Profit_margin
            1,        # Discount_flag
            1,        # Has_Feedback
            encoding_dict['Country']['USA'],
            encoding_dict['State']['California'],
            encoding_dict['City']['Los Angeles'],
            encoding_dict['Region']['West'],
            encoding_dict['Segment']['Consumer'],
            encoding_dict['Ship Mode']['Standard Class'],
            encoding_dict['Category']['Furniture'],
            encoding_dict['Sub-Category']['Chairs']
        ]
    }

    html_content = f"""
    <html>
        <head>
            <title>API Документация</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>API с моделью предсказания прибыли</h1>
            <h2>Кодировки категориальных признаков</h2>
            <pre>{json.dumps(encoding_dict, ensure_ascii=False, indent=4)}</pre>

            <h2>Пример POST запроса на /predict</h2>
            <pre>
POST /predict
Content-Type: application/json

{json.dumps(example_post, ensure_ascii=False, indent=4)}
            </pre>

            <h2>Описание модели входных данных</h2>
            <p>JSON с ключом <code>features</code> — список числовых признаков в порядке:</p>
            <ul>
                <li>Sales, Quantity, Discount, Price_per_unit, Profit_margin, Discount_flag, Has_Feedback,</li>
                <li>Кодировка Country, State, City, Region, Segment, Ship Mode, Category, Sub-Category</li>
            </ul>
        </body>
    </html>
    """
    return html_content

from fastapi import Body

class ModelInput(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: ModelInput = Body(...)):
    input_array = np.array([data.features])
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}

@app.get("/encoding_info")
def encoding_info():
    return encoding_dict
