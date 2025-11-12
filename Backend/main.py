# Backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- FIX 1: ADD CORS MIDDLEWARE ---
origins = [
    "http://127.0.0.1:5500",  
    "http://localhost:5500",
    "http://localhost:3000",   
    "http://127.0.0.1:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API is running"}


@app.post("/predict")
def predict_news(text: str):
    # In the future, we will load the model here and predict
    # result = model.predict(text)
    return {"result": "This functionality is coming soon!"}