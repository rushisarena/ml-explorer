"""
============================================================
ML Explorer — FastAPI Backend
============================================================
REST API for the Interactive ML Model Explorer.
Handles file upload, EDA, model training, SHAP explanations,
and prediction downloads.
============================================================
"""

import io
import uuid
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ml_engine import DataProcessor, ModelTrainer, ModelExplainer

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml-explorer")

# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="ML Explorer API",
    description="Interactive Machine Learning Model Explorer",
    version="1.0.0",
)

# CORS — allow frontend on different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SESSION STORAGE (in-memory, keyed by session ID)
# ============================================================
sessions: dict[str, dict] = {}
SESSION_TTL = 3600  # 1 hour


def cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["created_at"] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]
        log.info(f"Cleaned up session {sid}")


def get_session(session_id: str) -> dict:
    """Get session or raise 404."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")
    return sessions[session_id]


# ============================================================
# SAMPLE DATASETS
# ============================================================
SAMPLE_DIR = Path(__file__).parent / "sample_data"


def load_sample_dataset(name: str) -> bytes:
    """Load a built-in sample dataset."""
    path = SAMPLE_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample dataset '{name}' not found")
    return path.read_bytes()


# ============================================================
# PYDANTIC MODELS
# ============================================================
class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    test_size: float = 0.2
    models: list[str] | None = None


class ExplainRequest(BaseModel):
    session_id: str
    max_samples: int = 100


class PredictRequest(BaseModel):
    session_id: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "healthy", "sessions_active": len(sessions)}


@app.get("/api/sample-datasets")
async def list_sample_datasets():
    """List available sample datasets."""
    datasets = []
    if SAMPLE_DIR.exists():
        for f in SAMPLE_DIR.glob("*.csv"):
            df = pd.read_csv(f, nrows=2)
            datasets.append({
                "name": f.stem,
                "filename": f.name,
                "columns": len(df.columns),
            })
    return {"datasets": datasets}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(None), sample: str = None):
    """Upload a CSV file or load a sample dataset."""
    cleanup_sessions()

    if sample:
        content = load_sample_dataset(sample)
        filename = f"{sample}.csv"
    elif file:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        content = await file.read()
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="Provide a file or sample name")

    # Create session
    session_id = str(uuid.uuid4())[:8]
    processor = DataProcessor()

    try:
        schema = processor.load_csv(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    sessions[session_id] = {
        "processor": processor,
        "trainer": None,
        "explainer": None,
        "created_at": time.time(),
        "filename": filename,
    }

    log.info(f"Session {session_id}: uploaded {filename} ({schema['shape']['rows']} rows × {schema['shape']['cols']} cols)")

    return {
        "session_id": session_id,
        "filename": filename,
        "schema": schema,
    }


@app.post("/api/eda")
async def run_eda(request: dict):
    """Generate exploratory data analysis."""
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    session = get_session(session_id)
    processor: DataProcessor = session["processor"]

    try:
        eda = processor.generate_eda()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")

    return eda


@app.post("/api/train")
async def train_models(request: TrainRequest):
    """Train selected ML models."""
    session = get_session(request.session_id)
    processor: DataProcessor = session["processor"]

    log.info(f"Session {request.session_id}: training with target='{request.target_column}'")

    try:
        X_train, X_test, y_train, y_test = processor.prepare_data(
            target_col=request.target_column,
            test_size=request.test_size,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preparation failed: {str(e)}")

    trainer = ModelTrainer(processor.preprocessor, processor.task_type)

    try:
        results = trainer.train_all(X_train, X_test, y_train, y_test, request.models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    # Store for later use
    session["trainer"] = trainer
    session["X_test"] = X_test
    session["y_test"] = y_test

    log.info(f"Session {request.session_id}: trained {len(trainer.trained_models)} models, best={trainer.best_model_name}")

    return results


@app.post("/api/explain")
async def explain_model(request: ExplainRequest):
    """Generate SHAP explanations for the best model."""
    session = get_session(request.session_id)
    trainer = session.get("trainer")
    X_test = session.get("X_test")

    if trainer is None:
        raise HTTPException(status_code=400, detail="No models trained yet. Train models first.")

    explainer = ModelExplainer(trainer)

    try:
        explanations = explainer.explain(X_test, max_samples=request.max_samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

    return explanations


@app.post("/api/predict")
async def predict_download(request: PredictRequest):
    """Generate predictions and return as CSV download."""
    session = get_session(request.session_id)
    trainer = session.get("trainer")
    processor: DataProcessor = session["processor"]

    if trainer is None:
        raise HTTPException(status_code=400, detail="No models trained yet.")

    try:
        df = processor.original_df.copy()
        X = df.drop(columns=[processor.target_col], errors="ignore")

        # Use only the features that were used in training
        X_subset = X[processor.feature_names] if processor.feature_names else X

        predictions = trainer.predict(X_subset)

        df["predicted"] = predictions
        if hasattr(processor, "_label_encoder") and processor._label_encoder is not None:
            try:
                df["predicted_label"] = processor._label_encoder.inverse_transform(
                    predictions.astype(int)
                )
            except Exception:
                pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Return as CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{request.session_id}.csv"},
    )


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
