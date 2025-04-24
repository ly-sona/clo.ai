from pydantic import BaseModel, Field, HttpUrl
from typing import List

class JobCreated(BaseModel):
    job_id: str

class LayoutMeta(BaseModel):
    id: str
    thumb: HttpUrl
    power: float

class LayoutDetail(LayoutMeta):
    wns: float | None = None
    cells: int | None = None
    fullPng: HttpUrl | None = None