from util import SynaptrixClient
import pandas as pd
import time

client = SynaptrixClient(API_KEY="UToQoP0C-kzst9eTa-P9VhoHXf-j8Di39Uv", base_url="https://neurodiffusionapi-dev2.azurewebsites.net")
data = pd.read_csv("src/Synaptrix/dataset_1.csv")

start_time = time.time()
client.plot_denoised(data, data_columns=[1])
end_time = time.time()
time_elapsed=end_time-start_time
print(time_elapsed)
print()