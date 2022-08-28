from datetime import datetime
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../src')) # add edgeml to path

import time as timelib
import pandas as pd
from edgeml.predictor import Predictor, PredictorError

from model_python import score

p = Predictor(
    lambda input: score(input),
    ['ACC_x', 'ACC_y', 'GYRO_x', 'ACC_z', 'GYRO_y', 'GYRO_z'],
    50,
    ['6278fb975ebd3c0013327e63', '6278fb975ebd3c0013327e64']
)

test = pd.read_csv('./test.csv', index_col=0)

for t, row in test.iterrows():
    time = int(t) / 1e3
    for key, valStr in row.iteritems():
        p.add_datapoint(key, float(valStr), time)
    
    try:
        print(datetime.fromtimestamp(time), p.predict())
        timelib.sleep(0.08)
    except PredictorError:
        pass