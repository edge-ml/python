import tempfile
import h5py
import numpy as np
from edgeml.TimeSeries import TimeSeries
from edgeml.Labeling import Labeling, Label
from functools import reduce
import pandas as pd 

class Dataset():
    def __init__(self, backendURL, readKey=None, writeKey=None):
        self._backendURL = backendURL
        self._readKey = readKey
        self._writeKey = writeKey
        
        self._id = None
        self.name = None
        self.metaData = None
        self.timeSeries = None
        self.labelings = None

    def parse(self, data, labelings):
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
    def data(self):
        df = reduce(lambda x,y: pd.merge(x,y, on='time', how='outer'), [x.data for x in self.timeSeries])
        for labeling in self.labelings:
            for label in labeling.labels:
                if labeling.name not in df.columns:
                    df[labeling.name] = ""
                df.loc[(df['time'] >= label.start) & (df['time'] <= label.end), labeling.name] = label.name
                



        return df

    def loadData(self):
        for ts in self.timeSeries:
            ts.loadData()
