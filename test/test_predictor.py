from io import StringIO
import unittest
import time
from src.edgeml.predictor import Predictor, PredictorError
from pandas.util.testing import assert_frame_equal
import pandas as pd

class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self._time = 1652044007265

        self._p = Predictor(
            lambda input: 5,
            ['ACC_RAW_z', 'ACC_RAW_y', 'ACC_RAW_x'],
            5,
            ['62536901f2e49867b6936d9b', '62536901f2e49867b6936d9c']
        )

        self._z = [   5, None, None, None, None, None, None,   45, None,   49]
        self._y = [None, None,    9,   10,   11,   12, None,    1, None,   2]
        self._x = [   5,    6, None,   41,   42,   43,   44,   45,   46, None]

    def test_end(self):
        i = 0

        while i < len(self._x):
            t = self._time + i * 1000
            self._z[i] and self._p.add_datapoint('ACC_RAW_z', float(self._z[i]), t)
            self._y[i] and self._p.add_datapoint('ACC_RAW_y', float(self._y[i]), t)
            self._x[i] and self._p.add_datapoint('ACC_RAW_x', float(self._x[i]), t)
            i = i+1
        
        samples = Predictor._merge(self._p.store)
        interp = Predictor._interpolate(samples)

        assert_frame_equal(pd.read_csv(StringIO(interp.to_csv())), pd.read_csv(StringIO("""
,ACC_RAW_z,ACC_RAW_y,ACC_RAW_x,id
2022-05-08 21:06:47.265,5.0,9.0,5.0,0
2022-05-08 21:06:48.265,10.714285714285715,9.0,6.0,0
2022-05-08 21:06:49.265,16.42857142857143,9.0,23.5,0
2022-05-08 21:06:50.265,22.142857142857142,10.0,41.0,0
2022-05-08 21:06:51.265,27.857142857142858,11.0,42.0,0
2022-05-08 21:06:52.265,33.57142857142857,12.0,43.0,0
2022-05-08 21:06:53.265,39.285714285714285,6.5,44.0,0
2022-05-08 21:06:54.265,45.0,1.0,45.0,0
2022-05-08 21:06:55.265,47.0,1.5,46.0,0
2022-05-08 21:06:56.265,49.0,2.0,46.0,0""")))

    def test_before_window(self):
        i = 0

        while i < 4:
            t = self._time + i * 1000
            self._z[i] and self._p.add_datapoint('ACC_RAW_z', float(self._z[i]), t)
            self._y[i] and self._p.add_datapoint('ACC_RAW_y', float(self._y[i]), t)
            self._x[i] and self._p.add_datapoint('ACC_RAW_x', float(self._x[i]), t)
            i = i+1
        
        samples = Predictor._merge(self._p.store)
        interp = Predictor._interpolate(samples)

        assert_frame_equal(pd.read_csv(StringIO(interp.to_csv())), pd.read_csv(StringIO("""
,ACC_RAW_z,ACC_RAW_y,ACC_RAW_x,id
2022-05-08 21:06:47.265,5.0,9.0,5.0,0
2022-05-08 21:06:48.265,5.0,9.0,6.0,0
2022-05-08 21:06:49.265,5.0,9.0,23.5,0
2022-05-08 21:06:50.265,5.0,10.0,41.0,0""")))

    def test_after_window(self):
        i = 0

        while i < 7:
            t = self._time + i * 1000
            self._z[i] and self._p.add_datapoint('ACC_RAW_z', float(self._z[i]), t)
            self._y[i] and self._p.add_datapoint('ACC_RAW_y', float(self._y[i]), t)
            self._x[i] and self._p.add_datapoint('ACC_RAW_x', float(self._x[i]), t)
            i = i+1
        
        samples = Predictor._merge(self._p.store)
        interp = Predictor._interpolate(samples)

        assert_frame_equal(pd.read_csv(StringIO(interp.to_csv())), pd.read_csv(StringIO("""
,ACC_RAW_z,ACC_RAW_y,ACC_RAW_x,id
2022-05-08 21:06:47.265,5.0,9.0,5.0,0
2022-05-08 21:06:48.265,5.0,9.0,6.0,0
2022-05-08 21:06:49.265,5.0,9.0,23.5,0
2022-05-08 21:06:50.265,5.0,10.0,41.0,0
2022-05-08 21:06:51.265,5.0,11.0,42.0,0
2022-05-08 21:06:52.265,5.0,12.0,43.0,0
2022-05-08 21:06:53.265,5.0,12.0,44.0,0""")))

if __name__ == '__main__':
    unittest.main()