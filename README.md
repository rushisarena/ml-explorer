# 🔬 ML Model Explorer

An interactive, full-stack machine learning web application that lets users upload datasets, train multiple ML models, compare performance, and explore SHAP-based model explanations — all from the browser.

## 🎯 Features

- **📂 Data Upload**: Drag-and-drop CSV upload or use built-in sample datasets (Iris, Titanic, Housing)
- **🔍 Auto EDA**: Distribution histograms, correlation heatmaps, missing value analysis
- **⚙️ Multi-Model Training**: Train 5 ML models simultaneously (Logistic/Linear Regression, Random Forest, XGBoost, KNN, SVM)
- **📊 Model Comparison**: Side-by-side metrics, bar charts, radar plots, confusion matrices
- **🧠 SHAP Explanations**: Global feature importance, individual prediction explanations
- **⬇️ Download**: Export predictions as CSV

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS)  ←→  Backend (FastAPI + Python)
Chart.js + Vanilla JS        scikit-learn + XGBoost + SHAP
```

## 🚀 Quick Start

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

# Generate sample datasets
python generate_samples.py

# Start server
uvicorn main:app --reload --port 8000
```

### Frontend
Open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500`

## 📊 Supported Models

| Model | Classification | Regression |
|---|---|---|
| Logistic/Linear Regression | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| K-Nearest Neighbors | ✅ | ✅ |
| Support Vector Machine | ✅ | ✅ |

## 🔧 Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, XGBoost, SHAP, pandas, numpy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Design**: Glassmorphism, dark theme, responsive

## 📦 Deployment

### Render (Backend)
1. Push to GitHub
2. Connect repo to [Render](https://render.com)
3. Deploy as Web Service with `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend
Deploy `frontend/` to GitHub Pages, Vercel, or Netlify (static site).

## 👤 Author

Rushikesh Dhote

## 📄 License

MIT
