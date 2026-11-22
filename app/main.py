import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import config
from .registry import list_algorithms
from .runner import execute_run
from .schemas import PLOT_KINDS
from .store import read_status, run_dir, write_status

UUID_RE = re.compile(r"^[0-9a-f]{32}$")


def check_key(x_api_key: str = Header(default="")):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


api = FastAPI(title="AutoML API", dependencies=[Depends(check_key)])


def _valid_id(run_id: str) -> str:
    if not UUID_RE.match(run_id):
        raise HTTPException(404, "Unknown run")
    return run_id


@api.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


@api.get("/algorithms")
def algorithms():
    return {"algorithms": list_algorithms(), "plot_kinds": list(PLOT_KINDS)}


@api.post("/runs", status_code=202)
async def create_run(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    algo: str = Form(...),
    plots: str = Form("hist,count,heatmap_corr"),
    target: Optional[str] = Form(None),
    test_size: float = Form(0.2),
    callback_url: Optional[str] = Form(None),
):
    algos = list_algorithms()
    if algo not in algos:
        raise HTTPException(422, f"Unknown algo '{algo}'. Choose from {sorted(algos)}")
    plot_list = [p.strip() for p in plots.split(",") if p.strip()]
    bad = [p for p in plot_list if p not in PLOT_KINDS]
    if bad:
        raise HTTPException(422, f"Unknown plot kinds {bad}. Choose from {list(PLOT_KINDS)}")
    if not 0.05 <= test_size <= 0.5:
        raise HTTPException(422, "test_size must be between 0.05 and 0.5")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(422, "Only .csv files are supported")

    run_id = uuid.uuid4().hex
    path = run_dir(run_id) / "data.csv"
    limit, size = config.MAX_UPLOAD_MB * 1024 * 1024, 0
    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
            f.write(chunk)

    write_status(run_id, run_id=run_id, status="queued", stage="queued",
                 algo=algo, plots=plot_list, filename=file.filename)
    background.add_task(execute_run, run_id, str(path), algo, plot_list,
                        (target or "").strip() or None, test_size, callback_url or None)
    return {"run_id": run_id, "status": "queued"}


@api.get("/runs/{run_id}")
def get_run(run_id: str):
    s = read_status(_valid_id(run_id))
    if s is None:
        raise HTTPException(404, "Unknown run")
    return s


def _artifact(run_id: str, *parts: str) -> Path:
    p = run_dir(_valid_id(run_id)).joinpath(*parts)
    if not p.is_file():
        raise HTTPException(404, "Artifact not found (is the run finished?)")
    return p


@api.get("/runs/{run_id}/report")
def get_report(run_id: str):
    return FileResponse(_artifact(run_id, "report.html"), media_type="text/html",
                        filename="report.html", content_disposition_type="inline")


@api.get("/runs/{run_id}/plots/{name}")
def get_plot(run_id: str, name: str):
    return FileResponse(_artifact(run_id, "plots", Path(name).name), media_type="image/png")


@api.get("/runs/{run_id}/model")
def get_model(run_id: str):
    return FileResponse(_artifact(run_id, "model.joblib"), filename="model.joblib",
                        media_type="application/octet-stream")
