from util import SynaptrixClient
from example_data import single_segment_list, single_segment_array, single_segment_df, multi_segment_list, multi_segment_array, multi_segment_df, multi_channel_list, multi_channel_array, multi_channel_df
import numpy as np
multi_segment_csv = "multi_segment.csv"
multi_channel_csv = "multi_channel.csv"

api = "rinklybrain"

client = SynaptrixClient(api_key = api)

if __name__ == "__main__":
        
    
    denoised_single_data = client.denoise_segment(
        eeg_segment= single_segment_df, 
        output_format="df"
    )
    print("Denoised Data:")
    print(denoised_single_data)
    

    denoised_batch_data = client.denoise_batch(
        data_in= multi_segment_csv,
        num_channels = 1,
        sample_rate=512,
        output_format="list"
    )
    print("Denoised Batch data:")
    print(denoised_batch_data)

    denoised_batch_data_multi = client.denoise_batch(
        data_in= multi_channel_array,
        num_channels = 4,
        sample_rate=512,
        output_format="csv",
        file_name = "denoised_batch_multi.csv",
    )
    print("Denoised multichannel Batch data:")
    print(denoised_batch_data_multi, np.shape(denoised_batch_data_multi))
    
    plot = client.plot_denoised(
        data_in= multi_channel_csv,
        num_channels = 4,
        sample_rate=512,
        initial_window_sec=0.5
    )
    print(plot)
    
    rrmse = client.find_rrmse(
        eeg_segment = single_segment_list,
    )
    
    print(f'The RRMSE is {rrmse}')
    
    lsl_output = client.lsl_denoise(
        stream_duration = 0,
        num_channels = 4,
        sample_rate = 200,
        output_format = "csv",
        file_name = "lsl_test.csv"
        )

    print(lsl_output)