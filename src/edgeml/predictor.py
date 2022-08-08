from collections import deque
from functools import reduce
import time as timelib
import pandas as pd
import tsfresh

STORE_MAX_DATAPOINT_FACTOR = 10

class PredictorError(Exception):
    pass

class Predictor():
    def __init__(self, predictor, sensors, window_size, labels, scaler = None, windowing_mode = "sample"):
        self.predictor = predictor
        self.sensors = sensors
        self.window_size = window_size
        self.labels = labels
        self.scaler = scaler
        self.windowing_mode = windowing_mode
        self.last_add_time = None

        self.store = { key: {
            'data': deque(maxlen=self.window_size * STORE_MAX_DATAPOINT_FACTOR),
            'time': deque(maxlen=self.window_size * STORE_MAX_DATAPOINT_FACTOR)
        } for key in self.sensors }

    def add_datapoint(self, sensor_name: str, value: float, time: int = None):
        if (type(value) is not float):
            raise ValueError("Datapoint is not a number")
        
        if sensor_name not in self.sensors:
            raise ValueError("Sensor is not valid")
        
        if time is None:
            time = timelib.time()

        self.last_add_time = time
        self.store[sensor_name]["data"].append(value)
        self.store[sensor_name]["time"].append(round(time * 1000))

    def predict(self):
        samples = Predictor._merge(self.store)
        interpolated = Predictor._interpolate(samples)
        window = None
        if self.windowing_mode == 'sample':
            window = interpolated.tail(self.window_size)
            if len(window.index) < self.window_size:
                raise PredictorError("Not enough samples")
        elif self.windowing_mode == 'time':
            window = interpolated.last(str(self.window_size) + 'ms')
            # this is probably not ideal, with edge-fel NaN values don't throw an error, so js code doesn't drop these
            # still, this should only be a problem in the beginning phase since we do interpolation beforehand
            window = window.dropna(axis=0)
            if len(window.index) <= 0:
                raise PredictorError("Empty window")

        settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
        features = tsfresh.extract_features(
            window, column_id="id", default_fc_parameters=settings, n_jobs=0, disable_progressbar=True
        )
        l = features.iloc[0].values.tolist()

        if self.scaler is not None:
            for i in range(len(l)):
                l[i] = (l[i] - self.scaler["center"][i]) / self.scaler["scale"][i]
        
        pred = self.predictor(l)

        return {
            'prediction': self.labels[pred.index(max(pred))],
            'result': pred
        }

    @staticmethod
    def _merge(store):
        # python reduce is different, starts with [0] as acc and [1] as cur, no initial value but collection
        samples = reduce(
            lambda acc, cur: pd.merge(acc, cur, left_index=True, right_index=True, how="outer"), [
                pd.DataFrame(
                    data=list(value["data"]),
                    index=pd.to_datetime(list(value["time"]), unit="ms"),
                    columns=[key]
                ) for key, value in store.items()
            ]
        )
        samples["id"] = 0 # tsfresh needs an id column even when we are only interested in a single window, so stub it
        return samples
    
    @staticmethod
    def _interpolate(samples):
        return samples.interpolate(method="linear", limit_direction="both")