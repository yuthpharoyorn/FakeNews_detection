from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



origins = [
    "http://127.0.0.1:5500",  
    "http://localhost:5500",
    "http://localhost:3000",   
    "http://127.0.0.1:3000",  
    "file://", 
]

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Testing API endpoint"}

