import time
from edgeml.predictor import Predictor, PredictorError

from qeasdas_python import score

p = Predictor(
    lambda input: score(input),
    ['ACC_RAW_z', 'ACC_RAW_y', 'ACC_RAW_x'],
    100,
    ['62536901f2e49867b6936d9b', '62536901f2e49867b6936d9c']
)

while True:
	p.add_datapoint('ACC_RAW_z', getACC_RAW_z())
	p.add_datapoint('ACC_RAW_y', getACC_RAW_y())
	p.add_datapoint('ACC_RAW_x', getACC_RAW_x())

    try:
        print(p.predict())
    except PredictorError:
        pass
    
    time.sleep(0.25)