import asyncio
import time
import uuid

import pytest

from edgeml import edgeml


@pytest.mark.integration
def test_dataset_receiver_raises_on_invalid_key(backend_url):
    with pytest.raises(RuntimeError, match="Invalid key"):
        edgeml.DatasetReceiver(backend_url, readKey="bad")


@pytest.mark.integration
def test_dataset_receiver_loads_datasets(backend_url, external_api_keys):
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
        for idx in range(2):
            await collector.addDataPoint(timestamp + idx * 10, "Acc", float(idx))
        assert await collector.onCompleteAsync() is True

    asyncio.run(upload())

    receiver = edgeml.DatasetReceiver(backend_url, read_key, write_key)
    receiver.loadData()
    dataset = next(item for item in receiver.datasets if item.name == dataset_name)
    assert dataset.timeSeries[0].length == 2
    assert receiver.data
