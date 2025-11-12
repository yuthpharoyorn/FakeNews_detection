

# 📰 Fake News Link Detector API

## 📝 Project Description

This project provides a reliable API endpoint for validating the credibility of news article links. The frontend (HTML/CSS/JS) allows users to paste a URL, which is sent to this FastAPI backend. The backend extracts the article text and runs it through a trained NLP (Natural Language Processing) machine learning model to classify the content as **GENUINE** or **FAKE NEWS**.

### Key Technologies

  * **Backend API:** **FastAPI** (Python) + **Uvicorn** (ASGI Server)
  * **Machine Learning:** **Scikit-learn** (or TensorFlow/PyTorch) for NLP classification.
  * **Frontend:** **Vanilla HTML, CSS, JavaScript**
  * **Deployment:** CORS enabled for local testing.

-----

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  * **Python 3.9+**
  * **git**

### 1\. Clone the Repository

```bash
git clone https://github.com/yuthpharoyorn/fake-news-detector-frontend.git
cd fake-news-detector-frontend
```

### 2\. Set Up the Python Backend

This project uses a virtual environment (`.venv`) for dependency isolation.

```bash
# Create the virtual environment
python -m venv .venv

# Activate the environment
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install all necessary Python dependencies (FastAPI, ML libraries, etc.)
pip install -r backend/requirements.txt
```

### 3\. Acquire Model Artifacts (Crucial Step)

The trained machine learning model and vectorizer files are **not committed** to the repository (due to size). You must run the training script or download the artifacts:

  * **Option A (Recommended):** Run the training script.
    ```bash
    # Run the script to train the model and save artifacts to the 'models/' folder
    python path/to/your/train_model.py 
    ```
  * **Option B (If provided):** Download from a separate link:
      * [Link to Model File]
      * [Link to Vectorizer File]
      * Place both files (`fake_news_model.pkl` and `tfidf_vectorizer.pkl`) into the **`models/`** directory.

### 4\. Run the FastAPI Server

Start the API server on port 8000:

```bash

uvicorn backend.main:app --reload
```

The API is now running at `http://127.0.0.1:8000`.
you can also run on your local host on the specify port (check main.py)

-----

## 🧪 Testing the API

### 1\. Interactive Documentation (Swagger UI)

You can test the API directly without the frontend using the automatic documentation provided by FastAPI:

  * Open your browser to: `http://127.0.0.1:8000/docs`

### 2\. Frontend Integration

1.  Navigate to the `frontend/` directory.
2.  Open `index.html` in your web browser (or use an extension like Live Server, which typically runs on `http://127.0.0.1:5500`).
3.  Paste a link and click **Analyze**. The JavaScript will send the request to the FastAPI endpoint at port 8000.

-----

## 🔗 API Endpoints

| Method | Path | Description | Data Input | Data Output |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/` | Health check / Welcome message. | None | `{"message": "..."}` |
| **POST** | `/api/detect/` | **Main Prediction Endpoint.** Analyzes a given URL. | JSON: `{"url": "string"}` | JSON: `{"is_fake": boolean}` |

-----


## 📜 License

This project is licensed under the MIT License.
# Fake_News_Detection