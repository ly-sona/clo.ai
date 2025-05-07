from pydantic import BaseModel, Field
from typing import List

class JobCreated(BaseModel):
    job_id: str

class LayoutMeta(BaseModel):
    id: str
    thumb: str
    power: float
    original_power: float | None = None
    original_delay: float | None = None
    new_delay: float | None = None
    optimized_circuit: str | None = None

class LayoutDetail(LayoutMeta):
    wns: float | None = None
    cells: int | None = None
    fullPng: str | None = None