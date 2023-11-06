
#con2
import os
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from keras.models import load_model
import base64

import cv2
import numpy as np

# model = pickle.load(open(os.getcwd()+r'../model/final_model.pkl', 'rb'))
# model = pickle.load(open(os.path.join(os.getcwd(), '../model', 'Model-Flower.pkl'), 'rb'))
# model = load_model('D:\AI303\AnimalClasss\model\project_model.h5')
model = load_model('/CarBrandClass/model/project_model.h5')
# model_path = os.path.abspath(os.path.join(os.getcwd(), '../model/final_model.pkl'))
# model = pickle.load(open(model_path, 'rb'))


# สร้าง class เพื่อแปลงจาก json ข้อมูลมาเป็น object
class ImageClass(BaseModel):
    image_base64: str

app = FastAPI()
class_names = ['CAT','DOG','ELEPHANT','HORSE']

@app.get("/")
def read_root():
    return {"Class Brand Animal"}

@app.post("/api/animal") 
async def read_image(image: Request):
    image_base64_json = await image.json()
    image_data = base64.b64decode(image_base64_json['image_base64'])
    image_array = np.frombuffer(image_data, np.uint8) #binary data เป็น NumPyarray
    image_array = cv2.resize(image_array, (32, 32))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    yhat = model.predict(np.expand_dims(image_array, axis=0)) 
    print(f'Predicted: class={np.argmax(yhat)}')
    animal=class_names[np.argmax(yhat)]
    
    return {"animal is" : animal}
