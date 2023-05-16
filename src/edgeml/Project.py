import requests as req
from edgeml.consts import getProjectEndpoint
from edgeml.Dataset import Dataset


class Project():

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
        self.labeligns = res_data["labelings"]
        for d in datasets:
            tmp_dataset = Dataset(backendURL, self._readKey, self._writeKey)
            tmp_dataset.parse(d, self.labeligns)
            self.datasets.append(tmp_dataset)

    def loadData(self):
        for d in self.datasets:
            d.loadData()