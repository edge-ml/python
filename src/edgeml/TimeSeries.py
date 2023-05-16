import tempfile
import h5py
import numpy as np
from edgeml.consts import getProjectEndpoint
import requests as req
import pandas as pd

class SamplingRate:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var


class TimeSeries:
    def __init__(self, backendURL, datasetId, readKey=None, writeKey=None):
        self._backendURL = backendURL
        self._datasetId = datasetId
        self._readKey = readKey
        self._writeKey = writeKey
        self._id = None
        self.name = None
        self.start = None
        self.end = None
        self.unit = None
        self._data = None
        self.samplingRate = None
        self.length = None

    def parse(self, data):
        self._id = data["_id"]
        self.name = data["name"]
        self.start = data["start"]
        self.end = data["end"]
        self.unit = data["unit"]
        if data["samplingRate"] is not None:
            self.samplingRate = SamplingRate(data["samplingRate"]["mean"], data["samplingRate"]["var"])
        self.length = data["length"]

    @property
    def data(self):
        if self._data is None:
            raise Exception("You need to load the data first. Call loadData on the project, dataset, or time-series level.")
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def loadData(self) -> pd.DataFrame:
        res = req.get(self._backendURL + getProjectEndpoint + self._readKey + "/" + self._datasetId + "/" + self._id)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            temp_file.write(res.content)
            temp_file.flush()
            if self.length == 0 or self.length == None:
                self.data = pd.DataFrame(columns=['time', self.name])
            else:
                with h5py.File(temp_file.name, "r") as hf:
                    time_array = np.array(hf["time"])
                    data_array = np.array(hf["data"])
                    df = pd.DataFrame({"time": time_array, self.name: data_array})
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    self.data = df
