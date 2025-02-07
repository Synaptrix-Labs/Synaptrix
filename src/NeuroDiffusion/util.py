import numpy as np
import requests
import pandas as pd
from example_data import single_segment_list, single_segment_array, single_segment_df, multi_segment_list, multi_segment_array, multi_segment_df, multi_channel_list, multi_channel_array, multi_channel_df
multi_segment_csv = "multi_segment.csv"
multi_channel_csv = "multi_channel.csv"

api = "rinklybrain"
bad_api = "smoothbrain"


def convert_output(
    data: np.ndarray,
    num_channels: int = 1, 
    output_format: str = "array", 
    file_name: str = "denoised.csv"
):
    """
    Internal helper function to convert a NumPy array `data` 
    to the user-requested format: array, list, dataframe, or csv.
    
    - `data` is shape (channels, samples).
    - `output_format` can be "array", "list", "df", or "csv".
    - `file_name` can be used if you want to save CSV to disk. 
    """
    
    # array
    if output_format.lower() == "array":
        return data 
    
    # list
    elif output_format.lower() == "list":
        return data.tolist()
    
    # df
    elif output_format.lower() == "df":
        columns = [f"channel_{i+1}" for i in range(num_channels)]
        data_T = data.T
        df = pd.DataFrame(data_T, columns=columns)
        return df
    
    #csv
    elif output_format.lower() == "csv":
        columns = [f"channel_{i+1}" for i in range(num_channels)]
        data_T = data.T
        df = pd.DataFrame(data_T, columns=columns)
        df.to_csv(file_name, index=False, header=True)
        return file_name
    


def denoise_segment(
    eeg_segment: np.ndarray, 
    api_key: str,
    base_url: str = "http://localhost:8000",
    output_format: str = "array",
    file_name: str = "denoised.csv"
):
    """
    Denoise a single 512-sample segment (single channel) using the remote API.
    
    :param eeg_segment: 1D numpy array of shape (512,) representing a single-channel EEG.
    :param api_key: API key for authentication.
    :param base_url: Base URL of the FastAPI server.
    :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
    :param file_name: Used if output_format='csv'.
    """
    # Convert to list for JSON if needed
    if isinstance(eeg_segment, np.ndarray):
        eeg_segment_list = eeg_segment.tolist()
    elif isinstance(eeg_segment, pd.DataFrame):
        eeg_segment_list = eeg_segment[0].tolist()
    else:
        eeg_segment_list = eeg_segment
    
    
    try:
        response = requests.post(
            f"{base_url}/denoise",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={"noisy_eeg": eeg_segment_list}
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")
    
    denoised_list = response.json()["denoised_eeg"]
    denoised_array = np.array(denoised_list, dtype=np.float32)
    #print(f'the denoised array is {denoised_array}')    
    return convert_output(denoised_array, num_channels = 1, output_format = output_format, file_name = file_name)



def denoise_batch(
    data_in,
    api_key: str,
    num_channels: int = 1,
    sample_rate: int = 512,
    output_format: str = "array",
    file_name: str = "denoised_batch.csv",
    base_url: str = "http://localhost:8000"
):
    windows = []
    if num_channels == 1:
        if isinstance(data_in, np.ndarray):
            data_in_list = data_in.tolist()
        elif isinstance(data_in, pd.DataFrame):
            data_in_list = data_in.iloc[:, 0].tolist()
        elif isinstance(data_in, str):
            data_csv = pd.read_csv(data_in)
            data_in_list = data_csv.iloc[:, 0].tolist()
        else:
            data_in_list = data_in
        
        #if data is not divisible by sample rate trim data so that it is
        data_in_length = len(data_in_list)
        if data_in_length % sample_rate != 0:
            caboose = data_in_length % sample_rate
            data_in_list = data_in_list[:caboose]
        
        #pack data into nested list of dim (num seconds, data per second)    
        num_windows = int(len(data_in_list) / sample_rate)
        for i in range(num_windows):
            start = i * sample_rate
            end = start + sample_rate
            window = data_in_list[start:end]
            windows.append(window)

        #send to endpoint
        try:
            response = requests.post(
            "http://localhost:8000/batch-denoise",
                headers={
                    "x-api-key": "rinklybrain",
                    "Content-Type": "application/json"
                },
                json={
                    "noisy_eeg_batch": windows
                }
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")
    
        
        denoised_list = response.json()["denoised_eeg_batch"] # dimensions(num of seconds, data per second)
        print(np.shape(denoised_list))
        flat_denoised_list = list(np.concatenate(denoised_list))
        denoised_array = np.array(flat_denoised_list, dtype=np.float32)  # array of len (num of seconds * data per second)
        return convert_output(denoised_array, num_channels = 1, output_format = output_format, file_name = file_name)
    
    else:
        if isinstance(data_in, np.ndarray):
            data_in_list = data_in.tolist()
        elif isinstance(data_in, pd.DataFrame):
            data_in_list = data_in.values.T.tolist()
        elif isinstance(data_in, str):
            data_csv = pd.read_csv(data_in)
            data_in_list = data_csv.values.T.tolist()
        else:
            data_in_list = data_in
        
        #will store each denoised channel into this array
        denoised_array = []
        
        #check if each channel is a multiple of sampling rate
        data_in_length = np.shape(data_in_list)[1]
        for chan in data_in_list:
            windows = []
            if data_in_length % sample_rate != 0:
                caboose = data_in_length % sample_rate
                chan = chan[:caboose]
            
            #chunk data from each channel by the sampling rate and append to windows
            num_windows = int(data_in_length / sample_rate)
            for i in range(num_windows):
                start = i * sample_rate
                end = start + sample_rate
                window = chan[start:end]
                windows.append(window)
            
            #send each channel into batch denoise
            try:
                response = requests.post(
                "http://localhost:8000/batch-denoise",
                    headers={
                        "x-api-key": "rinklybrain",
                        "Content-Type": "application/json"
                    },
                    json={
                        "noisy_eeg_batch": windows
                    }
                )
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Request failed: {e}")
            denoised_chan_list = response.json()["denoised_eeg_batch"]
            flat_denoised_list = list(np.concatenate(denoised_chan_list))
            denoised_chan = np.array(flat_denoised_list, dtype=np.float32)
            denoised_array.append(denoised_chan)
            
        denoised_array = np.stack(denoised_array, axis=0) 
        return convert_output(denoised_array, num_channels = num_channels, output_format = output_format, file_name = file_name)
    

if __name__ == "__main__":
    
    api_key = "rinklybrain"
    
    # denoised_single_data = denoise_segment(
    #     eeg_segment= single_segment_df, 
    #     api_key=api_key, 
    #     output_format="csv"
    # )
    # print("Denoised Data:")
    # print(denoised_single_data)
    

    # denoised_batch_data = denoise_batch(
    #     data_in= multi_segment_csv,
    #     api_key=api_key,
    #     num_channels = 1,
    #     sample_rate=512,
    #     output_format="csv"
    # )
    # print("Denoised Batch data:")
    # print(denoised_batch_data)

    denoised_batch_data_multi = denoise_batch(
        data_in= multi_channel_csv,
        api_key=api_key,
        num_channels = 4,
        sample_rate=512,
        output_format="csv",
        file_name = "denoised_batch_multi.csv",
    )
    print("Denoised multichannel Batch data:")
    print(denoised_batch_data_multi, np.shape(denoised_batch_data_multi))