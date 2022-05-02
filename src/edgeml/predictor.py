from collections import deque
from functools import reduce
import time as timelib
import pandas as pd
import tsfresh

class PredictorError(Exception):
    pass

class Predictor():
    def __init__(self, predictor, sensors, window_size, labels):
        self.predictor = predictor
        self.sensors = sensors
        self.window_size = window_size
        self.labels = labels # TODO: we don't store this in the ml backend currently afaik

        self.store = { key: {
            'data': deque(maxlen=self.window_size * 2),
            'time': deque(maxlen=self.window_size * 2)
        } for key in self.sensors }

    def add_datapoint(self, sensor_name: str, value: float, time: int = None):
        if (type(value) is not float):
            raise ValueError("Datapoint is not a number")
        
        if sensor_name not in self.sensors:
            raise ValueError("Sensor is not valid")
        
        if time is None:
            time = round(timelib.time() * 1000)

        self.store[sensor_name]["data"].append(value)
        self.store[sensor_name]["time"].append(time)

    def predict(self):
        # python reduce is different, starts with [0] as acc and [1] as cur, no initial value but collection
        samples = reduce(
            lambda acc, cur: pd.merge(acc, cur, left_index=True, right_index=True, how="outer"), [
                pd.DataFrame(
                    data=list(value["data"]),
                    index=pd.to_datetime(list(value["time"]), unit="ms"),
                    columns=[key]
                ) for key, value in self.store.items()
            ]
        )
        samples["id"] = 0 # tsfresh needs an id column even when we are only interested in a single window, so stub it

        interpolated = samples.interpolate(method="linear", limit_direction="both")

        window = interpolated.tail(self.window_size)

        if (len(window.index) < self.window_size):
            raise PredictorError("Not enough samples")

        settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
        features = tsfresh.extract_features(
            window, column_id="id", default_fc_parameters=settings, n_jobs=0
        )

        l = features.iloc[0].values.tolist()
        pred = self.predictor(l)

        return pred # TODO: use labels to map labels