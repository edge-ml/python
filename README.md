# Obtain data from edge-ml


```python
from edgeml import DatasetReceiver
```

### 1. Create a project
This will also pull the metadata from the server.
Here you need the *read*-key.
Make sure there is not trailing */* in your URL.


```python
project = DatasetReceiver("https://edge-ml-beta.dmz.teco.edu", "8c051972b56e6b4ad6bd0bf573da580f")
print(project)
```

    Dataset - Name: W_001, ID: 645a255bdd19d537f2a50126, Metadata: {}
    Dataset - Name: W_004, ID: 645a255b7d9569d03843a12b, Metadata: {}
    Dataset - Name: Square_003, ID: 645a255b2ce253c26b52426c, Metadata: {}
    Dataset - Name: Square_004, ID: 645a255b1d6af5e7ed04575f, Metadata: {}
    Dataset - Name: Square_002, ID: 645a255bc21261d4cbdc990d, Metadata: {}
    Dataset - Name: W_002, ID: 645a255b3ebe0af02b211d5d, Metadata: {}
    Dataset - Name: W_003, ID: 645a255b074737af782167e2, Metadata: {}
    Dataset - Name: Square_001, ID: 645a255bbacf5ab8ff044304, Metadata: {}
    Dataset - Name: edgemlDemo, ID: 64635a79a7a7513e8ac92d32, Metadata: {'langauge': 'python'}
    Dataset - Name: edgemlDemo, ID: 64635a7b6fec1d2800b1cd06, Metadata: {'langauge': 'python'}


### 2. Actually obtain data
Until now, we only have metadata available. We need to pull the actual time-series data using one of the following methods:

*We can load the data for a single timeSeries*


```python

project.datasets[0].timeSeries[0].loadData()
project.datasets[0].timeSeries[0].data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01 00:10:23.339</td>
      <td>-823.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-01-01 00:10:23.378</td>
      <td>-819.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-01-01 00:10:23.418</td>
      <td>-770.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-01-01 00:10:23.458</td>
      <td>-746.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-01-01 00:10:23.497</td>
      <td>-783.0</td>
    </tr>
  </tbody>
</table>
</div>



*For a single dataset*


```python
project.datasets[0].loadData()
project.datasets[0].data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>Gestures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01 00:10:23.339</td>
      <td>-823.0</td>
      <td>-45.0</td>
      <td>4025.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-01-01 00:10:23.378</td>
      <td>-819.0</td>
      <td>-158.0</td>
      <td>4075.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-01-01 00:10:23.418</td>
      <td>-770.0</td>
      <td>-255.0</td>
      <td>4116.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-01-01 00:10:23.458</td>
      <td>-746.0</td>
      <td>-155.0</td>
      <td>4059.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-01-01 00:10:23.497</td>
      <td>-783.0</td>
      <td>-104.0</td>
      <td>3963.0</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



*Or for all datasets in the project*


```python
project.loadData()
project.data[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>Gestures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01 00:10:23.339</td>
      <td>-823.0</td>
      <td>-45.0</td>
      <td>4025.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-01-01 00:10:23.378</td>
      <td>-819.0</td>
      <td>-158.0</td>
      <td>4075.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-01-01 00:10:23.418</td>
      <td>-770.0</td>
      <td>-255.0</td>
      <td>4116.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-01-01 00:10:23.458</td>
      <td>-746.0</td>
      <td>-155.0</td>
      <td>4059.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-01-01 00:10:23.497</td>
      <td>-783.0</td>
      <td>-104.0</td>
      <td>3963.0</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



# Upload data to edge-ml

### 1. Upload data using timestamps from the device
Here you will need the *write*-key


```python
from edgeml.edgeml import DatasetCollector
sender = DatasetCollector(url="https://edge-ml-beta.dmz.teco.edu", apiKey="4e6159c9c77124d71f298e93f1ed7254", name="edgemlDemo", useDeviceTime=True, timeSeries=["accX", "accY"], metaData={"langauge": "python"})

```


```python
import time

for i in range (100):
    await sender.addDataPoint(name="accX", value=i*0.1)
    await sender.addDataPoint(name="accY", value=i*0.5)
    time.sleep(0.01)
sender.onComplete()
```




    True



### 2. Provide your own timestamps


```python
from edgeml.edgeml import DatasetCollector
sender = DatasetCollector(url="https://edge-ml-beta.dmz.teco.edu", apiKey="4e6159c9c77124d71f298e93f1ed7254", name="edgemlDemo", useDeviceTime=False, timeSeries=["accX", "accY"], metaData={"langauge": "python"})

```


```python
import time

for i in range (100):
    await sender.addDataPoint(timestamp=i*1000, name="accX", value=i*0.1)
    await sender.addDataPoint(timestamp=i*1000, name="accY", value=i*0.5)
    time.sleep(0.01)
sender.onComplete()
```




    True


