from edgeml import DatasetReceiver




WRITE_KEY = "399b57a1a98c6f7450ef8235f086a815"
READ_KEY = "0f1a0ea264a33383969f21b4b45b82ad"

localUrl = "https://beta.edge-ml.org"
datasetName = "Example Dataset"

project = DatasetReceiver(localUrl, READ_KEY, WRITE_KEY)

print(project)

project.loadData()

print(project.datasets[0].timeSeries[0].data.head())

print(project.data)