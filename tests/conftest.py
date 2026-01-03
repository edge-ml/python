from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIMEOUT = 10


@pytest.fixture(scope="session")
def backend_url():
    return os.getenv("BACKEND_URL", "http://backend:8000")


@pytest.fixture(scope="session")
def backend_ready(backend_url):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            res = requests.get(f"{backend_url}/docs", timeout=2)
            if res.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("Backend did not become ready in time")


@pytest.fixture
def access_token(backend_ready, backend_url):
    suffix = uuid.uuid4().hex[:8]
    username = f"test_{suffix}"
    email = f"{username}@example.com"
    password = "testpassword"

    res = requests.post(
        f"{backend_url}/api/v1/auth/register",
        data={"username": username, "email": email, "password": password},
        timeout=TIMEOUT,
    )
    if res.status_code != 201:
        raise RuntimeError(f"Registration failed: {res.status_code} {res.text}")

    res = requests.post(
        f"{backend_url}/api/v1/auth/token",
        data={"username": username, "password": password},
        timeout=TIMEOUT,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Login failed: {res.status_code} {res.text}")
    data = res.json()
    return data["access_token"]


@pytest.fixture
def project_id(access_token, backend_url):
    res = requests.post(
        f"{backend_url}/api/v1/projects/",
        json={"name": f"python-it-{uuid.uuid4().hex[:6]}"},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=TIMEOUT,
    )
    if res.status_code != 201:
        raise RuntimeError(f"Project creation failed: {res.status_code} {res.text}")
    return res.json()["id"]


@pytest.fixture
def external_api_keys(project_id, access_token, backend_url):
    res = requests.post(
        f"{backend_url}/api/v1/{project_id}/external_api/switch/true",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=TIMEOUT,
    )
    if res.status_code not in (200, 201):
        raise RuntimeError(f"Enable external api failed: {res.status_code} {res.text}")

    res = requests.put(
        f"{backend_url}/api/v1/{project_id}/external_api/",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=TIMEOUT,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Key generation failed: {res.status_code} {res.text}")
    data = res.json()
    return data["read_key"], data["write_key"]
