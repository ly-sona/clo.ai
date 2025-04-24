"""
Ultra-light ‘DB’ so we don’t pull in Redis for a prototype.
In production swap this for Postgres or RedisJSON.
"""
from datetime import datetime
from typing import Dict, Any

_jobs: Dict[str, Dict[str, Any]] = {}

def create(job_id: str, **fields):
    _jobs[job_id] = {"created": datetime.utcnow(), **fields}

def get(job_id: str) -> dict | None:
    return _jobs.get(job_id)

def update(job_id: str, **patch):
    if job_id in _jobs:
        _jobs[job_id].update(patch)

def list_all() -> list[dict]:
    return list(_jobs.values())