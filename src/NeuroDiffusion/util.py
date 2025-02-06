import numpy as np
import requests
from example_data import single_segment_data, multi_segment_data

api = "rinklybrain"
bad_api = "smoothbrain"


def denoise_segment(data, api):
    
    response = requests.post(
        "http://localhost:8000/denoise",
        headers={
            "x-api-key": api,
            "Content-Type": "application/json"
        },
        json={
            "noisy_eeg": data
        }
    )

    status = response.status_code
    print("\nstatus:", status)
    print("DENOISE")
    if status == 200:
        result = response.json()
        print(result)
    else:
        return print("error:", response.json())
        

def denoise_batch(data, api):
    data = np.array(data)
    window_size = 512
    num_windows = len(data) // window_size
    windows = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = data[start:end].tolist()
        windows.append(window)
        
    response = requests.post(
    "http://localhost:8000/batch-denoise",
    headers={
        "x-api-key": api,
        "Content-Type": "application/json"
    },
    json={
        "noisy_eeg_batch": windows
    }
    )

    status = response.status_code
    print("status:", status)
    print("BATCH DENOISE")
    if status == 200:
        result = response.json()
        print(result)
    else:
        print("error:", response.json())
        

data = single_segment_data
denoise_segment(data, api)

data = multi_segment_data
denoise_batch(data, api)