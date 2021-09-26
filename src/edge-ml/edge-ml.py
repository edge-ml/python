import requests

def data(url, api_key):
    project = requests.post(url, { key: api_key })
    print(project)