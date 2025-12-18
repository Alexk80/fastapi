from datetime import date
from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
app = FastAPI()

# Загрузка модели из файла pickle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('samples.csv', 'r') as file:
    data = pd.read_csv(file)
    

with open('scaler_42.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
# Счетчик запросов
request_count = 0

# Модель для валидации входных данных
class PredictionInput(BaseModel):
    number: int

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/samples")
def sample():
    return {"request_count": [data.iloc[0]]}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    
    '''new_data = pd.DataFrame({
        'Pclass': [input_data.Pclass],
        'Age': [input_data.Age],
        'Fare': [input_data.Fare]
    })'''

    # Предсказание
    predictions = model.predict([data.iloc[input_data.number]])

    # Преобразование результата в человеко-читаемый формат
    result = "Пользователь совершил действие" if predictions[0] == 1 else "Пользователь не совершал действий"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)