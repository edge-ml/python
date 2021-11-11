import requests as req
import time as timelib


uploadDataset = "/api/deviceApi/uploadDataset"
initDatasetIncrement = "/api/deviceApi/initDatasetIncrement"
addDatasetIncrement = "/api/deviceApi/addDatasetIncrement"
addDatasetIncrementBatch = "/api/deviceApi/addDatasetIncrementBatch"

#
#  Uploads a whole dataset to a specific project
#  @param {string} url - The url of the backend server
#  @param {string} key - The Device-Api-Key
#  @param {object} dataset - The dataset to upload
#  @returns A Promise indicating success or failure
#

def sendDataset(url: str, key: str, dataset: dict):
    try:
        res = req.post(url + uploadDataset, json = {"key": key, "payload": dataset})
    except req.exceptions.RequestException:
        raise "error" #TODO
    

#
#  @param {string} url - The url of the backend server
#  @param {string} key - The Device-Api-Key
#  @param {boolean} useDeviceTime - True if you want to use timestamps generated by the server
#  @returns Function to upload single datapoints to one dataset inside a specific project
#
class datasetCollector():
    def __init__(self, url: str, key: str, name: str, useDeviceTime: bool) -> None:
        self.url = url
        self.key = key
        self.name = name
        self.useDeviceTime = useDeviceTime

        res = req.post(url + initDatasetIncrement, json = {"deviceApiKey": key, "name": name})
        #TODO error handling
        self.datasetKey = res.json()['datasetKey']
        self.dataStore = {'datasetKey': self.datasetKey, 'data': []}
        self.counter = 0
        self.error = None
    

    def addDataPoint(self, sensorName: str, value: float, time: int = None):
        if (self.error):
            raise self.error
        if (type(value) is not float): #TODO cast int to float, it may cause problems
            raise ValueError("Datapoint is not a number")
        if (not self.useDeviceTime and type(time) is not int):
            raise ValueError("Provide a valid timestamp")
        
        if (self.useDeviceTime):
            time = timelib.time()

        if (all(dataPoint['sensorname'] != sensorName for dataPoint in self.dataStore['data'])):
            self.dataStore['data'].append({
                'sensorname': sensorName, #TODO sensorname is not in camelcase, maybe refactor later in db?
                'start': time,
                'end': time,
                'timeSeriesData': [{'timestamp': time, 'datapoint': value}]
            })
        else:
            for dataPoint in self.dataStore['data']:
                if (dataPoint['sensorname'] == sensorName):
                    dataPoint['timeSeriesData'].append({'timestamp': time, 'datapoint': value})
                    dataPoint['start'] = min(dataPoint['start'], time)
                    dataPoint['end'] = max(dataPoint['end'], time)
                    break
        
        self.counter = self.counter + 1
        if self.counter > 1000:
            self.upload()

    def __upload(self):
        res = req.post(self.url + addDatasetIncrementBatch, json = self.dataStore)
        self.counter = 0
        self.dataStore = {'datasetKey': self.datasetKey, 'data': []}

    def onComplete(self):
        if self.error:
            raise self.error
        self.__upload()