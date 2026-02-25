from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, PositiveInt, conlist, conint

app = FastAPI(title="API")

@dataclass
class Project:
    id: int
    name: str
    description: str = ""

@dataclass
class Dataset:
    id: int
    project_id: int
    name: str
    values: list[float]


_projects: dict[int, Project] = {}
_datasets: dict[int, Dataset] = {}

_project_id_seq = 1
_dataset_id_seq = 1


Horizon = conint(ge=1, le=10000)

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""

class ProjectUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""

class ProjectOut(BaseModel):
    id: int
    name: str
    description: str

class DatasetCreate(BaseModel):
    project_id: PositiveInt
    name: str = Field(..., min_length=1)
    values: conlist(float, min_items=2)

class DatasetUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    values: conlist(float, min_items=2)

class DatasetOut(BaseModel):
    id: int
    project_id: int
    name: str
    values: list[float]

def naive_forecast(values: list[float], horizon: int) -> list[float]:
    last = float(values[-1]) if values else 0.0
    return [last] * horizon

def mae(y_true: list[float], y_pred: list[float]) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    return sum(abs(float(y_true[i]) - float(y_pred[i])) for i in range(n)) / n

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/projects", response_model=ProjectOut, status_code=201)
def create_project(req: ProjectCreate) -> ProjectOut:
    global _project_id_seq
    pid = _project_id_seq
    _project_id_seq += 1
    p = Project(id=pid, name=req.name, description=req.description)
    _projects[pid] = p
    return ProjectOut(id=p.id, name=p.name, description=p.description)

@app.get("/api/projects", response_model=list[ProjectOut])
def list_projects() -> list[ProjectOut]:
    return [ProjectOut(id=p.id, name=p.name, description=p.description) for p in _projects.values()]

@app.get("/api/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: int) -> ProjectOut:
    p = _projects.get(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut(id=p.id, name=p.name, description=p.description)

@app.put("/api/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, req: ProjectUpdate) -> ProjectOut:
    p = _projects.get(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    p.name = req.name
    p.description = req.description
    _projects[project_id] = p
    return ProjectOut(id=p.id, name=p.name, description=p.description)

@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: int):
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")

    #каскадное удаление датасетов
    ds_to_delete = [d.id for d in _datasets.values() if d.project_id == project_id]
    for did in ds_to_delete:
        delete_dataset(did)

    del _projects[project_id]
    return None

# ---- Datasets ----
@app.post("/api/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(req: DatasetCreate) -> DatasetOut:
    global _dataset_id_seq
    if req.project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")

    did = _dataset_id_seq
    _dataset_id_seq += 1
    d = Dataset(id=did, project_id=req.project_id, name=req.name, values=req.values)
    _datasets[did] = d
    return DatasetOut(id=d.id, project_id=d.project_id, name=d.name, values=d.values)

@app.get("/api/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: Optional[int] = Query(default=None)) -> list[DatasetOut]:
    items = list(_datasets.values())
    if project_id is not None:
        items = [d for d in items if d.project_id == project_id]
    return [DatasetOut(id=d.id, project_id=d.project_id, name=d.name, values=d.values) for d in items]

@app.get("/api/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int) -> DatasetOut:
    d = _datasets.get(dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetOut(id=d.id, project_id=d.project_id, name=d.name, values=d.values)

@app.put("/api/datasets/{dataset_id}", response_model=DatasetOut)
def update_dataset(dataset_id: int, req: DatasetUpdate) -> DatasetOut:
    d = _datasets.get(dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    d.name = req.name
    d.values = req.values
    _datasets[dataset_id] = d
    return DatasetOut(id=d.id, project_id=d.project_id, name=d.name, values=d.values)

@app.delete("/api/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int):
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    del _datasets[dataset_id]
    return None
