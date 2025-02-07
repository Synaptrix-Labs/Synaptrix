import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import seaborn as sns

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
    """
    Denoise a multi channel and time series as long as you want.
    
    :param data_in: array, list, df, or csv
    :param api_key: API key for authentication.
    :num_channels: how many channels in the data
    :sample_rate: how many data points per second
    :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
    :param file_name: Used if output_format='csv'.
    :param base_url: Base URL of the FastAPI server.
    """
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
                    "x-api-key": api_key,
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
                        "x-api-key": api_key,
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

sns.set_theme()

def plot_denoised(
    data_in, # shape (channels, samples)
    api_key: str,
    num_channels: int,
    sample_rate: int = 512,
    initial_window_sec: float = 2.0,
    base_url: str = "http://localhost:8000"
):
    """
    Create an interactive figure showing the clean and noisy time serives

    :param data_in: array, list, df, or csv
    :param api_key: for authenticating your API
    :param num_channels: number of EEG channels
    :param sample_rate: sampling rate in Hz
    :param initial_window_sec: initial view window width in seconds
    :param base_url: address of your FastAPI server
    """
    if isinstance(data_in, list):
        data_in_array = np.array(data_in)
    elif isinstance(data_in, pd.DataFrame):
        data_in_array = np.array(data_in.values.T)
    elif isinstance(data_in, str):
        data_csv = pd.read_csv(data_in)
        data_in_array = np.array(data_csv.values.T)
    else:        
        data_in_array = data_in
    #Call denoise_batch
    denoised_array = denoise_batch(
        data_in=data_in,
        api_key=api_key,
        num_channels=num_channels,
        sample_rate=sample_rate,
        output_format="array",
        base_url=base_url
    )
    channels, total_samples = denoised_array.shape
    
    # Ensure data_in_array has the same shape
    assert data_in_array.shape == (channels, total_samples), (
        "data_in_array shape must match the shape of denoised data."
    )

    # creat subplot for each channel
    fig, axes = plt.subplots(nrows=channels, ncols=1, sharex=True, figsize=(10, 6))
    if channels == 1:
        axes = [axes]

    fig.suptitle("NeuroDiffusion (Denoised & Noisy)", fontsize=14)
    time = np.arange(total_samples) / sample_rate  # shape => (total_samples,)

    # this tracks
    # - current start index
    # - current window width in samples
    start_idx = 0
    current_window_sec = initial_window_sec
    window_samples = int(current_window_sec * sample_rate)
    window_samples = min(window_samples, total_samples)

    end_idx = start_idx + window_samples

    # Plot lines for each channel
    denoised_lines = []
    noisy_lines = []
    for ch in range(channels):
        ax = axes[ch]

        # Denoised line
        den_line, = ax.plot(
            time[start_idx:end_idx],
            denoised_array[ch, start_idx:end_idx],
            color="C0", lw=1.2, label="Denoised"
        )
        denoised_lines.append(den_line)

        # Noisy line
        noisy_line, = ax.plot(
            time[start_idx:end_idx],
            data_in_array[ch, start_idx:end_idx],
            color="C1", lw=1.0, label="Noisy"
        )
        noisy_lines.append(noisy_line)

        ax.set_ylabel(f"Channel {ch+1}")


    axes[-1].set_xlabel("Time (sec)")
    if end_idx > 0:
        axes[-1].set_xlim(time[start_idx], time[end_idx-1])
    else:
        axes[-1].set_xlim(0, 0)

    # toggle noisy lines on/off
    show_noisy = False

    # 5) Update function to redraw lines based on current window
    def update_plot():
        nonlocal start_idx, end_idx
        end_idx = start_idx + window_samples
        if end_idx > total_samples:
            end_idx = total_samples
            start_idx = end_idx - window_samples

        for ch in range(channels):
            # Denoised
            denoised_lines[ch].set_xdata(time[start_idx:end_idx])
            denoised_lines[ch].set_ydata(denoised_array[ch, start_idx:end_idx])

            # Noisy
            noisy_lines[ch].set_xdata(time[start_idx:end_idx])
            noisy_lines[ch].set_ydata(data_in_array[ch, start_idx:end_idx])

        if end_idx > 0:
            axes[-1].set_xlim(time[start_idx], time[end_idx-1])
        else:
            axes[-1].set_xlim(0, 0)

        fig.canvas.draw_idle()

    # 6) Button callbacks (Left/Right):
    def on_left(event):
        nonlocal start_idx
        step = window_samples // 2 if window_samples > 1 else 1
        start_idx = max(0, start_idx - step)
        update_plot()

    def on_right(event):
        nonlocal start_idx
        step = window_samples // 2 if window_samples > 1 else 1
        start_idx = min(start_idx + step, total_samples - window_samples)
        update_plot()

    # update window width
    def on_window_change(text):
        nonlocal current_window_sec, window_samples
        try:
            val = float(text)
            if val <= 0:
                return
        except ValueError:
            return
        current_window_sec = val
        window_samples = int(current_window_sec * sample_rate)
        window_samples = max(1, min(window_samples, total_samples))
        update_plot()

    # show/hide noisy lines
    def on_toggle_noisy(event):
        nonlocal show_noisy
        show_noisy = not show_noisy
        for line in noisy_lines:
            line.set_visible(show_noisy)
        fig.canvas.draw_idle()

    # Place the buttons & text box on the figure
    ax_left = plt.axes([0.12, 0.01, 0.08, 0.05])
    ax_right = plt.axes([0.23, 0.01, 0.08, 0.05])
    ax_box = plt.axes([0.45, 0.01, 0.1, 0.05])
    ax_toggle = plt.axes([0.65, 0.01, 0.12, 0.05])

    btn_left = Button(ax_left, "Left")
    btn_right = Button(ax_right, "Right")
    text_box = TextBox(ax_box, "Window(sec):", initial=str(initial_window_sec))
    btn_toggle = Button(ax_toggle, "Toggle Noisy")

    # Link callbacks
    btn_left.on_clicked(on_left)
    btn_right.on_clicked(on_right)
    text_box.on_submit(on_window_change)
    btn_toggle.on_clicked(on_toggle_noisy)

    for line in noisy_lines:
        line.set_visible(show_noisy)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()

    
def find_rrmse(
    eeg_segment,
    api_key: str,
    base_url: str = "http://localhost:8000",
):
    """
    Find RRMSE for a segment of EEG data
    
    :param eeg_segment: array, list, or df
    :param api_key: API key for authentication.
    :param base_url: Base URL of the FastAPI server.
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
            f"{base_url}/denoise-rrmse",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "noisy_eeg": eeg_segment_list
            }
        )

        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")
        
    RRMSE = response.json()["rrmse"]
    return print(f"The RRMSE of this segment is {RRMSE}")



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

    # denoised_batch_data_multi = denoise_batch(
    #     data_in= multi_channel_array,
    #     api_key=api_key,
    #     num_channels = 4,
    #     sample_rate=512,
    #     output_format="array",
    #     file_name = "denoised_batch_multi.csv",
    # )
    # print("Denoised multichannel Batch data:")
    # print(denoised_batch_data_multi, np.shape(denoised_batch_data_multi))
    
    # plot = plot_denoised(
    #     data_in= multi_channel_csv,
    #     api_key=api_key,
    #     num_channels = 4,
    #     sample_rate=512,
    #     initial_window_sec=0.5
    # )
    # print(plot)
    
    find_rrmse(
        eeg_segment = single_segment_list,
        api_key=api_key,
    )
    
