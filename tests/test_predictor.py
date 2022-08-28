from io import StringIO
import unittest
import tsfresh
from src.edgeml.predictor import Predictor
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal
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

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        _time = 1652044007265

        _p = Predictor(
            lambda input: 5,
            ['ACC_RAW_z', 'ACC_RAW_y', 'ACC_RAW_x'],
            5,
            ['62536901f2e49867b6936d9b', '62536901f2e49867b6936d9c']
        )

        _z = [   5, None, None, None, None, None, None,   45, None,   49]
        _y = [None, None,    9,   10,   11,   12, None,    1, None,   2]
        _x = [   5,    6, None,   41,   42,   43,   44,   45,   46, None]

        i = 0

        while i < len(_x):
            t = _time + i * 1000
            _z[i] and _p.add_datapoint('ACC_RAW_z', float(_z[i]), t)
            _y[i] and _p.add_datapoint('ACC_RAW_y', float(_y[i]), t)
            _x[i] and _p.add_datapoint('ACC_RAW_x', float(_x[i]), t)
            i = i+1

        samples = Predictor._merge(_p.store)
        self.interp = Predictor._interpolate(samples)

    def test_tsfresh(self):
        settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
        features = tsfresh.extract_features(
            self.interp, column_id="id", default_fc_parameters=settings, n_jobs=0, disable_progressbar=True
        )

        # ordering
        self.assertEqual(features.iloc[0].index.tolist(), [
            'ACC_RAW_z__sum_values',
            'ACC_RAW_z__median',
            'ACC_RAW_z__mean',
            'ACC_RAW_z__length',
            'ACC_RAW_z__standard_deviation',
            'ACC_RAW_z__variance',
            'ACC_RAW_z__root_mean_square',
            'ACC_RAW_z__maximum',
            'ACC_RAW_z__absolute_maximum',
            'ACC_RAW_z__minimum', 'ACC_RAW_y__sum_values', 'ACC_RAW_y__median', 'ACC_RAW_y__mean', 'ACC_RAW_y__length', 'ACC_RAW_y__standard_deviation', 'ACC_RAW_y__variance', 'ACC_RAW_y__root_mean_square', 'ACC_RAW_y__maximum', 'ACC_RAW_y__absolute_maximum', 'ACC_RAW_y__minimum', 'ACC_RAW_x__sum_values', 'ACC_RAW_x__median', 'ACC_RAW_x__mean', 'ACC_RAW_x__length', 'ACC_RAW_x__standard_deviation', 'ACC_RAW_x__variance', 'ACC_RAW_x__root_mean_square', 'ACC_RAW_x__maximum', 'ACC_RAW_x__absolute_maximum', 'ACC_RAW_x__minimum'])
        # values
        assert_array_equal(features.iloc[0].values.tolist(), [
            296.0,
            30.714285714285715,
            29.6,
            10.0,
            14.89908913802643,
            221.9828571428571,
            33.13823859445244,
            49.0,
            49.0,
            5.0, 71.0, 9.0, 7.1, 10.0, 3.916631205513228, 15.34, 8.108637370113428, 12.0, 12.0, 1.0, 341.5, 42.5, 34.15, 10.0, 15.614176251086706, 243.80250000000007, 37.55029959933742, 46.0, 46.0, 5.0]
        ) 


if __name__ == '__main__':
    unittest.main()