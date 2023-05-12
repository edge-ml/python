import time
import requests

URLS = {
    "uploadDataset": "/api/deviceapi/uploadDataset",
    "initDatasetIncrement": "/ds/api/dataset/init/",
    "addDatasetIncrement": "/ds/api/dataset/append/",
}

UPLOAD_INTERVAL = 5 * 1000

class DatasetCollector:
    def __init__(
        self, url, apiKey, name, useDeviceTime, timeSeries, metaData, datasetLabel
    ):
        self.url = url
        self.apiKey = apiKey
        self.name = name
        self.useDeviceTime = useDeviceTime
        self.timeSeries = timeSeries
        self.metaData = metaData
        self.datasetLabel = datasetLabel
        self.error = None
        self.dataStore = {"data": []}
        self.uploadComplete = False
        self.labeling = None

        if self.datasetLabel:
            labeling_name, label_name = self.datasetLabel.split("_")
            self.labeling = {"labelingName": labeling_name, "labelName": label_name}

        self.dataStore = {"data": []}
        self.error = None

        res = requests.post(
            url + URLS["initDatasetIncrement"] + apiKey,
            json={
                "name": self.name,
                "metaData": self.metaData,
                "timeSeries": self.timeSeries,
                "labeling": self.labeling,
            },
        )

        if res.status_code != 200:
            raise RuntimeError(res.text.split(':')[1][1:-2])

        res_data = res.json()
        self.lastChecked = time.time()
        if not res_data or not res_data["id"]:
            raise RuntimeError("Could not generate DatasetCollector")
        self.datasetKey = res_data["id"]

    async def addDataPoint(self, timestamp, name, value):
        if name not in self.timeSeries:
            raise ValueError("invalid time-series name")

        if self.error:
            raise ValueError(self.error)

        if not isinstance(value, (int, float)):
            raise ValueError("Datapoint is not a number")

        if not self.useDeviceTime and not isinstance(timestamp, (int, float)):
            raise ValueError("Provide a valid timestamp")

        if self.useDeviceTime:
            timestamp = int(time.time() * 1000)

        value = round(value * 100) / 100
        
        if all(elm["name"] != name for elm in self.dataStore["data"]):
            self.dataStore["data"].append(
                {
                    "name": name,
                    "data": [[timestamp, value]],
                }
            )
        else:
            idx = next(
                (
                    idx
                    for idx, elm in enumerate(self.dataStore["data"])
                    if elm["name"] == name
                ),
                None,
            )
            self.dataStore["data"][idx]["data"].append([timestamp, value])
            if self.dataStore["data"][idx].get('start') is None or self.dataStore["data"][idx]["start"] > timestamp:
                self.dataStore["data"][idx]["start"] = timestamp
            if self.dataStore["data"][idx].get('end') is None or self.dataStore["data"][idx]["end"] < timestamp:
                self.dataStore["data"][idx]["end"] = timestamp

        if time.time() * 1000 - self.lastChecked > UPLOAD_INTERVAL:
            await self.upload(self.labeling)
            self.lastChecked = time.time() * 1000
            self.dataStore = {"data": []}

    async def upload(self, uploadLabel):
        tmp_dataStore = self.dataStore.copy()
        response = requests.post(
            self.url
            + URLS["addDatasetIncrement"]
            + self.apiKey
            + "/"
            + self.datasetKey,
            json={"data": tmp_dataStore["data"], "labeling": uploadLabel},
        )
        if response.status_code != 200:
            raise RuntimeError("Upload failed")

    # Synchronizes the server with the data when you have added all data
    async def onComplete(self):
        if self.uploadComplete:
            raise RuntimeError("Dataset is already uploaded")
        await self.upload(self.labeling)
        if self.error:
            raise RuntimeError(self.error)
        self.uploadComplete = True
        return True