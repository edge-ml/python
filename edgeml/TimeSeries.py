from __future__ import annotations

import io
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import requests as req

from edgeml.consts import getProjectEndpoint

class SamplingRate:
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var


class TimeSeries:
    def __init__(
        self,
        backendURL: str,
        datasetId: int,
        readKey: Optional[str] = None,
        writeKey: Optional[str] = None,
    ):
        self._backendURL = backendURL
        self._datasetId = datasetId
        self._readKey = readKey
        self._writeKey = writeKey
        self._id: Optional[int] = None
        self.name: Optional[str] = None
        self.start: Optional[int] = None
        self.end: Optional[int] = None
        self.unit: Optional[str] = None
        self._data: Optional[pd.DataFrame] = None
        self.samplingRate: Optional[SamplingRate] = None
        self.length: Optional[int] = None

    def parse(self, data: dict) -> None:
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
    def data(self, value: pd.DataFrame) -> None:
        self._data = value

    def loadData(self) -> pd.DataFrame:
        res = req.get(
            self._backendURL
            + getProjectEndpoint
            + self._readKey
            + "/"
            + str(self._datasetId)
            + "/"
            + str(self._id)
        )
        with io.BytesIO(res.content) as temp_file:
            if self.length == 0 or self.length == None:
                self.data = pd.DataFrame(columns=['time', self.name])
            else:
                with h5py.File(temp_file, "r") as hf:
                    time_array = np.array(hf["time"])
                    data_array = np.array(hf["data"])
                    df = pd.DataFrame({"time": time_array, self.name: data_array})
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    self.data = df
