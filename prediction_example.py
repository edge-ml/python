import time
from src.edgeml.predictor import Predictor
from urllib.request import urlopen
import pickle

model = pickle.loads(urlopen("http://localhost:3003/ml/deploy/e7029d31-d568-4042-ad46-8aaac459f174/export/python").read())

p = Predictor(None, ['AAA', 'BBB', 'CCC'], 3)

p.add_datapoint('AAA', 0.5)
p.add_datapoint('BBB', 10.5)
p.add_datapoint('CCC', 56.5)
time.sleep(0.1)
p.add_datapoint('AAA', 39.5)
p.add_datapoint('CCC', 19.5)
p.add_datapoint('BBB', 110.5)
time.sleep(0.1)
p.add_datapoint('AAA', 29.5)
time.sleep(0.1)
p.add_datapoint('CCC', 46.5)
time.sleep(0.1)
p.add_datapoint('AAA', 27.5)
p.add_datapoint('BBB', 74.3)
time.sleep(0.1)

p.predict()

