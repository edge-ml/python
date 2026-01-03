from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional

import pandas as pd

from edgeml.Labeling import Labeling
from edgeml.TimeSeries import TimeSeries

class Dataset():
    def __init__(self, backendURL: str, readKey: Optional[str] = None, writeKey: Optional[str] = None):
        self._backendURL = backendURL
        self._readKey = readKey
        self._writeKey = writeKey
        
        self._id: Optional[int] = None
        self.name: Optional[str] = None
        self.metaData: Optional[Dict[str, object]] = None
        self.timeSeries: Optional[List[TimeSeries]] = None
        self.labelings: Optional[List[Labeling]] = None

    def parse(self, data: Dict[str, object], labelings: List[Dict[str, object]]) -> None:
        self._id = data["_id"]
        self.name = data["name"]
        self.metaData = data["metaData"]
        self.timeSeries = []
        for ts in data["timeSeries"]:
            tmp_timeSeries = TimeSeries(self._backendURL, self._id, self._readKey, self._writeKey)
            tmp_timeSeries.parse(ts)
            self.timeSeries.append(tmp_timeSeries)

        self.labelings = []
        label_name_map =  {label['_id']: label['name'] for entry in labelings for label in entry.get('labels', [])}

        for labeling in data["labelings"]:
            labeling["name"] = next(x["name"] for x in labelings if x["_id"] == labeling["labelingId"])
            for label in labeling["labels"]:
                label["name"] = label_name_map[label["type"]]
            temp_labeling = Labeling()
            temp_labeling.parse(labeling)
            self.labelings.append(temp_labeling)

    @property
    def data(self) -> pd.DataFrame:
        df = reduce(lambda x,y: pd.merge(x,y, on='time', how='outer'), [x.data for x in self.timeSeries])
        for labeling in self.labelings:
            for label in labeling.labels:
                if labeling.name not in df.columns:
                    df[labeling.name] = ""
                if label.start < 0 or label.start > 2147483647000 or label.end < 0 or label.end > 2147483647000:
                    continue
                label_start = pd.to_datetime(label.start, unit='ms')
                label_end = pd.to_datetime(label.end, unit='ms')
                df.loc[(df['time'] >= label_start) & (df['time'] <= label_end), labeling.name] = label.name
        return df

    def loadData(self) -> None:
        for ts in self.timeSeries:
            ts.loadData()


    def __str__(self) -> str:
        return f"Dataset - Name: {self.name}, ID: {self._id}, Metadata: {self.metaData}"
