import edgeml
import time
from tqdm.auto import tqdm


WRITE_KEY = "399b57a1a98c6f7450ef8235f086a815"
localUrl = "https://beta.edge-ml.org"
datasetName = "Example Dataset"

import math
import asyncio

timeSeries = ["Acc", "Mag"]
metaData = {}

async def main():
    print("Creating DatasetCollector")
    collector = edgeml.DatasetCollector(localUrl,
                                        WRITE_KEY,
                                        datasetName,
                                        False,
                                        timeSeries,
                                        metaData)

    print("DatasetCollector created")

    timestamp = round(time.time() * 1000)
    for i in tqdm(range(10)):
        timestamp += 40
        x = i / 10000        # Adjust the divisor to control the frequency of the wave
        y_acc = math.sin(x)  # Generate the y-coordinate for "Acc"
        y_mag = math.cos(x)  # Generate the y-coordinate for "Mag"
        await collector.addDataPoint(timestamp, "Acc", y_acc)
        await collector.addDataPoint(timestamp, "Mag", y_mag)

    # signal data collection is complete
    collector.onComplete()


print("Starting data collection")
asyncio.run(main())
