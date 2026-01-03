import os
import time
import uuid

import asyncio
import requests

from edgeml import edgeml


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
TIMEOUT = 10


def wait_for_backend():
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            res = requests.get(f"{BACKEND_URL}/docs", timeout=2)
            if res.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("Backend did not become ready in time")


def register_and_login():
    suffix = uuid.uuid4().hex[:8]
    username = f"test_{suffix}"
    email = f"{username}@example.com"
    password = "testpassword"

    res = requests.post(
        f"{BACKEND_URL}/api/v1/auth/register",
        data={"username": username, "email": email, "password": password},
        timeout=TIMEOUT,
    )
    if res.status_code != 201:
        raise RuntimeError(f"Registration failed: {res.status_code} {res.text}")

    res = requests.post(
        f"{BACKEND_URL}/api/v1/auth/token",
        data={"username": username, "password": password},
        timeout=TIMEOUT,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Login failed: {res.status_code} {res.text}")
    data = res.json()
    return data["access_token"]


def create_project(access_token):
    res = requests.post(
        f"{BACKEND_URL}/api/v1/projects/",
        json={"name": f"python-it-{uuid.uuid4().hex[:6]}"},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=TIMEOUT,
    )
    if res.status_code != 201:
        raise RuntimeError(f"Project creation failed: {res.status_code} {res.text}")
    return res.json()["id"]


def enable_external_api(project_id, access_token):
    res = requests.post(
        f"{BACKEND_URL}/api/v1/{project_id}/external_api/switch/true",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=TIMEOUT,
    )
    if res.status_code not in (200, 201):
        raise RuntimeError(f"Enable external api failed: {res.status_code} {res.text}")

    res = requests.put(
        f"{BACKEND_URL}/api/v1/{project_id}/external_api/",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=TIMEOUT,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Key generation failed: {res.status_code} {res.text}")
    data = res.json()
    return data["read_key"], data["write_key"]


def test_device_api_upload_and_download():
    wait_for_backend()
    access_token = register_and_login()
    project_id = create_project(access_token)
    read_key, write_key = enable_external_api(project_id, access_token)

    dataset_name = f"python-it-dataset-{uuid.uuid4().hex[:6]}"
    async def upload():
        collector = edgeml.DatasetCollector(
            BACKEND_URL,
            write_key,
            dataset_name,
            False,
            ["Acc"],
            {},
        )

        timestamp = int(time.time() * 1000)
        for idx in range(3):
            await collector.addDataPoint(timestamp + idx * 10, "Acc", float(idx))

        assert await collector.onCompleteAsync() is True

    asyncio.run(upload())

    receiver = edgeml.DatasetReceiver(BACKEND_URL, read_key)
    receiver.loadData()
    dataset = next(item for item in receiver.datasets if item.name == dataset_name)
    dataset.loadData()

    assert len(dataset.timeSeries) == 1
    ts = dataset.timeSeries[0]
    assert ts.name == "Acc"
    assert ts.length == 3

    df = dataset.data
    assert "Acc" in df.columns
    assert len(df.index) == 3
