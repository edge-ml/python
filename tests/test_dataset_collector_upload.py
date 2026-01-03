import asyncio
import time
import uuid

import pytest

from edgeml import edgeml


@pytest.mark.integration
@pytest.mark.integration
def test_add_datapoint_invalid_name(backend_url, external_api_keys):
    _, write_key = external_api_keys
    collector = edgeml.DatasetCollector(
        backend_url,
        write_key,
        f"dataset-{uuid.uuid4().hex[:6]}",
        False,
        ["Acc"],
        {},
    )

    with pytest.raises(ValueError, match="invalid time-series name"):
        asyncio.run(collector.addDataPoint(0, "Gyro", 1.0))


@pytest.mark.integration
def test_add_datapoint_invalid_value(backend_url, external_api_keys):
    _, write_key = external_api_keys
    collector = edgeml.DatasetCollector(
        backend_url,
        write_key,
        f"dataset-{uuid.uuid4().hex[:6]}",
        False,
        ["Acc"],
        {},
    )

    with pytest.raises(ValueError, match="Datapoint is not a number"):
        asyncio.run(collector.addDataPoint(0, "Acc", "bad"))


@pytest.mark.integration
def test_add_datapoint_invalid_timestamp(backend_url, external_api_keys):
    _, write_key = external_api_keys
    collector = edgeml.DatasetCollector(
        backend_url,
        write_key,
        f"dataset-{uuid.uuid4().hex[:6]}",
        False,
        ["Acc"],
        {},
    )

    with pytest.raises(ValueError, match="Timestamp is not an integer"):
        asyncio.run(collector.addDataPoint(1.5, "Acc", 1.0))


@pytest.mark.integration
def test_integration_upload_with_labeling(backend_url, external_api_keys):
    read_key, write_key = external_api_keys
    dataset_name = f"python-it-dataset-{uuid.uuid4().hex[:6]}"

    async def upload():
        collector = edgeml.DatasetCollector(
            backend_url,
            write_key,
            dataset_name,
            False,
            ["Acc"],
            {},
            datasetLabel="activity_walk",
        )
        timestamp = int(time.time() * 1000)
        for idx in range(3):
            await collector.addDataPoint(timestamp + idx * 10, "Acc", float(idx))
        assert await collector.onCompleteAsync() is True

    asyncio.run(upload())

    receiver = edgeml.DatasetReceiver(backend_url, read_key)
    receiver.loadData()
    dataset = next(item for item in receiver.datasets if item.name == dataset_name)
    assert len(dataset.labelings) == 1
    label = dataset.labelings[0].labels[0]
    assert dataset.labelings[0].name == "activity"
    assert label.name == "walk"


@pytest.mark.integration
def test_integration_upload_without_labeling(backend_url, external_api_keys):
    read_key, write_key = external_api_keys
    dataset_name = f"python-it-dataset-{uuid.uuid4().hex[:6]}"

    async def upload():
        collector = edgeml.DatasetCollector(
            backend_url,
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

    receiver = edgeml.DatasetReceiver(backend_url, read_key)
    dataset = next(item for item in receiver.datasets if item.name == dataset_name)
    assert dataset.labelings == []


@pytest.mark.integration
def test_upload_ignores_labeling(backend_url, external_api_keys):
    read_key, write_key = external_api_keys
    dataset_name = f"python-it-dataset-{uuid.uuid4().hex[:6]}"

    async def upload():
        collector = edgeml.DatasetCollector(
            backend_url,
            write_key,
            dataset_name,
            False,
            ["Acc"],
            {},
            datasetLabel="activity_walk",
        )
        await collector.addDataPoint(int(time.time() * 1000), "Acc", 1.0)
        await collector.upload(None)
        assert await collector.onCompleteAsync() is True

    asyncio.run(upload())

    receiver = edgeml.DatasetReceiver(backend_url, read_key)
    dataset = next(item for item in receiver.datasets if item.name == dataset_name)
    assert dataset.labelings == []
