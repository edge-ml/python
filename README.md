# edge-ml python
[![Tests](https://github.com/edge-ml/python/actions/workflows/tests.yml/badge.svg)](https://github.com/edge-ml/python/actions/workflows/tests.yml)
[![PyPI Publish](https://github.com/edge-ml/python/actions/workflows/PyPIPublish.yml/badge.svg)](https://github.com/edge-ml/python/actions/workflows/PyPIPublish.yml)

Python package for [edge-ml.org](https://edge-ml.org).

## Usage
### Installation
Install edge-ml using the follwing command.
```
pip install edge-ml
```

### Get Project
```python
import edgeml

# get the API key from the settings of your project
project = edgeml.getProject("https://app.edge-ml.org", PROJECT_API_KEY) 
```

## Development
### Testing
To run the tests please enter:
```
python -m unittest -v tests/all.py
```
