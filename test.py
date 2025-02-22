from util import SynaptrixClient
import pandas as pd

client = SynaptrixClient(API_KEY = "rinklybrain")

data = pd.read_csv("tests/dataset_1.csv")

#denoised = client.denoise_batch(data, data_columns=[1,2,3], skip_rows=151000, datetime_column=0, output_format="csv", file_name="nonono.csv")

plot = client.plot_denoised(data, data_columns=[1,5], skip_rows = 151000, sample_rate=256)

# lsl_output = client.lsl_denoise(
#     stream_duration = 20, # in seconds, change parameter to 0 for indefinite streaming
#     num_channels = 4,
#     sample_rate = 200, # adjust to match sampling rate of your device
#     output_format = "csv",
#     file_name = "lsl_test.csv" 
    
#     # at the conclusion of the stream, all denoised data will be saved to this file
# )