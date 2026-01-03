import requests as req
from .consts import getProjectEndpoint, initDatasetIncrement, addDatasetIncrement
from .Dataset import Dataset
import time
import asyncio

class DatasetReceiver:

    def __init__(self, backendURL, readKey=None, writeKey=None):
        self.backendURL = backendURL
        self._readKey=readKey
        self._writeKey=writeKey
        res = req.get(backendURL + getProjectEndpoint + readKey)
        if res.status_code == 403:
            raise RuntimeError("Invalid key")
        elif res.status_code >= 300:
            raise RuntimeError(res.reason)
        self.datasets = []
        res_data = res.json()
        datasets = res_data["datasets"]
        self.labelings = res_data["labelings"]
        for d in datasets:
            tmp_dataset = Dataset(backendURL, self._readKey, self._writeKey)
            tmp_dataset.parse(d, self.labelings)
            self.datasets.append(tmp_dataset)

    def loadData(self):
        for d in self.datasets:
            d.loadData()

    @property
    def data(self):
        return [x.data for x in self.datasets]

    def __str__(self) -> str:
        return "\n".join([str(x) for x in self.datasets])

URLS = {
    "uploadDataset": "/api/v1/deviceapi/uploadDataset",
    "initDatasetIncrement": "/api/v1/deviceapi/dataset/init/",
    "addDatasetIncrement": "/api/v1/deviceapi/dataset/append/",
    "getDatasetsInProject": "/api/v1/deviceapi/datasets/"
}

UPLOAD_INTERVAL = 5 * 1000

class DatasetCollector:
    def __init__(
        self, url, write_key, name, use_own_timestamps, timeSeries, metaData, datasetLabel=None
    ):
        self.url = url
        self.apiKey = write_key
        self.name = name
        self.use_own_timestamps = use_own_timestamps
        self.timeSeries = timeSeries
        self.metaData = metaData
        self.datasetLabel = datasetLabel
        self.error = None
        self.uploadComplete = False
        self.labeling = None
        self.lastChecked = time.time() * 1000
        self._pending_uploads = set()

        if self.use_own_timestamps:
            self.addDataPoint = self._addDataPoint_DeviceTime
        else:
            self.addDataPoint = self._addDataPoint_OwnTimeStamps


        if self.datasetLabel:
            labeling_name, label_name = self.datasetLabel.split("_")
            self.labeling = {"labelingName": labeling_name, "labelName": label_name}

        self.error = None

        res = req.post(
            url + initDatasetIncrement + self.apiKey,
            json={
                "name": self.name,
                "metaData": self.metaData,
                "timeSeries": self.timeSeries,
                "labeling": self.labeling,
            },
        )

        if res.status_code != 200:
            raise RuntimeError(res.text)

        res_data = res.json()
        if not res_data or not res_data["id"]:
            raise RuntimeError("Could not generate DatasetCollector")
        self.datasetKey = res_data["id"]
        self.dataStore = {x: [] for x in self.timeSeries}

    def _drain_data_store(self):
        snapshot = {k: v[:] for k, v in self.dataStore.items()}
        self.dataStore = {x: [] for x in self.timeSeries}
        return snapshot
    
    async def _addDataPoint_DeviceTime(self, name, value):
        timestamp = int(time.time() * 1000)
        await self._addDataPoint_OwnTimeStamps(timestamp, name, value)

    async def _addDataPoint_OwnTimeStamps(self, timestamp, name, value):
        if name not in self.timeSeries:
            raise ValueError("invalid time-series name")

        if not isinstance(value, (int, float)):
            raise ValueError("Datapoint is not a number")

        if not isinstance(timestamp, int):
            raise ValueError(f"Timestamp is not an integer: {timestamp}")
        self.dataStore[name].append([timestamp, value])

        if time.time() * 1000 - self.lastChecked > UPLOAD_INTERVAL:
            payload = self._drain_data_store()
            task = asyncio.create_task(self._upload_payload(payload, self.labeling))
            self._pending_uploads.add(task)
            task.add_done_callback(self._pending_uploads.discard)
            self.lastChecked = time.time() * 1000


    async def _upload_payload(self, payload, uploadLabel):
        tmp_dataStore = [{"name": k, "data": payload[k]} for k in payload.keys()]
        response = req.post(
            self.url
            + addDatasetIncrement
            + self.apiKey
            + "/"
            + self.datasetKey,
            json={"data": tmp_dataStore, "labeling": uploadLabel},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.status_code} {response.text}")

    async def upload(self, uploadLabel):
        payload = self._drain_data_store()
        await self._upload_payload(payload, None)

    async def _await_pending_uploads(self):
        if not self._pending_uploads:
            return
        pending = list(self._pending_uploads)
        self._pending_uploads.clear()
        await asyncio.gather(*pending)

    # Synchronizes the server with the data when you have added all data
    async def onCompleteAsync(self):
        if self.uploadComplete:
            raise RuntimeError("Dataset is already uploaded")
        await self._await_pending_uploads()
        payload = self._drain_data_store()
        if any(payload.values()):
            await self._upload_payload(payload, self.labeling)
        if self.error:
            raise RuntimeError(self.error)
        self.uploadComplete = True
        return True

    def onComplete(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.onCompleteAsync())
        raise RuntimeError("onComplete must be awaited; use await onCompleteAsync()")
