import time
from edgeml.predictor import Predictor, PredictorError
import pickle

model = # pickle.loads(...) # load model file with pickle

p = Predictor(
    None,
    ['AAA', 'BBB', 'CCC'],
    3
)

while True:
    p.add_datapoint('AAA', 0.5)
    p.add_datapoint('BBB', 10.5)
    p.add_datapoint('CCC', 56.5)

    try:
        print(p.predict())
    except PredictorError:
        pass
    
    time.sleep(0.25)