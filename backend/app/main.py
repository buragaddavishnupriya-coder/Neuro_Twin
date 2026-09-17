from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI(
    title="NeuroTwin API",
    description="Multi-Modal Digital Twin & Neurological Clinical Inference API for Parkinson's and Alzheimer's Diseases",
    version="1.0.0"
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

candidate_dirs = [
    Path(__file__).resolve().parent.parent / "artifacts" / "paper_results",
    Path(__file__).resolve().parents[2] / "artifacts" / "paper_results",
    Path(__file__).resolve().parent / "artifacts" / "paper_results",
]
ARTIFACTS_DIR = next((p for p in candidate_dirs if p.exists() and (p / "ieee_results.json").exists()), candidate_dirs[0])
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount paper results static directory so frontend can display plots directly
app.mount("/static/plots", StaticFiles(directory=str(ARTIFACTS_DIR)), name="plots")


@app.get("/")
def root():
    return {
        "project": "NeuroTwin",
        "status": "Running",
        "modules": [
            "Parkinson's Disease Clinical Model",
            "Alzheimer's Disease Clinical Model",
            "Brain MRI Diagnostic Swin Transformer",
            "Multi-Modal Digital Twin Engine",
            "IEEE Publication Results & Graph Generator",
        ],
    }


@app.get("/api/results/summary")
def get_results_summary():
    """Retrieve full IEEE publication checklist metrics and metadata."""
    json_path = ARTIFACTS_DIR / "ieee_results.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail="IEEE results have not been generated yet. Run generation first.",
        )
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


@app.get("/api/results/latex", response_class=PlainTextResponse)
def get_latex_table():
    """Retrieve publication-ready IEEE LaTeX table string."""
    tex_path = ARTIFACTS_DIR / "results_table.tex"
    if not tex_path.exists():
        raise HTTPException(
            status_code=404,
            detail="LaTeX table has not been generated yet.",
        )
    with open(tex_path, "r", encoding="utf-8") as fp:
        return fp.read()


@app.post("/api/results/regenerate")
def regenerate_results():
    """Trigger on-demand training and generation of paper figures & metrics."""
    from scripts.generate_paper_results import run_all_evaluations
    run_all_evaluations()
    return {"status": "Success", "message": "All IEEE paper results and graphs regenerated."}