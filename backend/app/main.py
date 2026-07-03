import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .qwen_client import qwen_configured
from .tools import (
    generate_report,
    load_experiment_data,
    rank_candidates,
    summarize_data,
    train_surrogate_model,
)


class AnalyzeResponse(BaseModel):
    summary: dict[str, Any]
    model_metrics: dict[str, Any]
    answer: str
    candidates: list[dict[str, Any]]
    provider: str
    warning: str | None = None


app = FastAPI(
    title="AI4S Qwen Thermite Few-shot Data Agent",
    description="Safe AI4S data analysis agent powered by Qwen OpenAI-compatible API.",
    version="0.1.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "qwen_configured": qwen_configured()}


@app.get("/api/demo-summary")
async def demo_summary() -> dict[str, Any]:
    df = load_experiment_data()
    model, metrics = train_surrogate_model(df)
    return {
        "summary": summarize_data(df),
        "model_metrics": metrics,
        "candidates": rank_candidates(df, model),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    question: str = Form(...),
    file: UploadFile | None = File(default=None),
) -> AnalyzeResponse:
    try:
        file_bytes = await file.read() if file else None
        df = load_experiment_data(
            file_bytes=file_bytes,
            filename=file.filename if file else None,
        )
        summary = summarize_data(df)
        model, metrics = train_surrogate_model(df)
        candidates = rank_candidates(df, model)
        report = await generate_report(question, summary, candidates, metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(
        summary=summary,
        model_metrics=metrics,
        answer=report["answer"],
        candidates=candidates,
        provider=report["provider"],
        warning=report.get("warning"),
    )

