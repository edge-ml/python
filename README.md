# edge-ml python
[![Tests](https://github.com/edge-ml/python/actions/workflows/tests.yml/badge.svg)](https://github.com/edge-ml/python/actions/workflows/tests.yml)
[![PyPI Publish](https://github.com/edge-ml/python/actions/workflows/PyPIPublish.yml/badge.svg)](https://github.com/edge-ml/python/actions/workflows/PyPIPublish.yml)

Python package for [edge-ml.org](https://edge-ml.org).

## Usage
### Installation
Install edge-ml using the follwing command.
```bash
pip install edge-ml
```

### Retrieve Project
This functionality is commonly applied if you would like to train a machine learning model from edge-ml.
```python
import edgeml

# get the API key from the settings of your project
project = edgeml.getProject("https://app.edge-ml.org", PROJECT_API_KEY) 
```

### Push Data from Python
```python
import edgeml
import time

key = "YOUR_API_KEY"
startTime = time.time()
collector = edgeml.datasetCollector("https://app.edge-ml.org",
                                    key,
                                    "Example Dataset", # name the dataset you would like to upload
                                    False) # do not use server timestamps

labels = [random.randint(1,3) for i in range(500)]
for i in range(500):
    collector.addDataPoint("Accelerometer", random.randint(1,50)/10.0, startTime + i) # adding samples at random time steps

collector.onComplete()
```

## Development
### Testing
To run the tests please enter:

```bash
python -m unittest -v tests/all.py
```
