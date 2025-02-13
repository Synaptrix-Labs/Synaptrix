import time
import math
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import seaborn as sns
from pylsl import StreamInlet, resolve_streams



class SynaptrixClient:
    def __init__(self, API_KEY: str, base_url: str = "https://neurodiffusionapi.azurewebsites.net"):
        self.API_KEY = API_KEY
        self.base_url = base_url


    def convert_output(
        self,
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
        self,
        eeg_segment: np.ndarray, 
        output_format: str = "array",
        file_name: str = "denoised.csv"
    ):
        """
        Denoise a single 512-sample segment (single channel) using the remote API.
        
        :param eeg_segment: 1D numpy array of shape (512,) representing a single-channel EEG.
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
                f"{self.base_url}/denoise",
                headers={
                    "x-api-key": self.API_KEY,
                    "Content-Type": "application/json"
                },
                json={"noisy_eeg": eeg_segment_list}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")
        
        denoised_list = response.json()["denoised_eeg"]
        denoised_array = np.array(denoised_list, dtype=np.float32)
        return self.convert_output(denoised_array, num_channels = 1, output_format = output_format, file_name = file_name)


    def denoise_batch(
        self,
        data_in,
        num_channels: int = 1,
        sample_rate: int = 512,
        output_format: str = "array",
        file_name: str = "denoised_batch.csv",
    ):
        """
        Denoise a multi channel and time series as long as you want.
        
        :param data_in: array, list, df, or csv
        :num_channels: how many channels in the data
        :sample_rate: how many data points per second
        :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
        :param file_name: Used if output_format='csv'.
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
                f"{self.base_url}/batch-denoise",
                    headers={
                        "x-api-key": self.API_KEY,
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
            return self.convert_output(denoised_array, num_channels = 1, output_format = output_format, file_name = file_name)
        
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
                    f"{self.base_url}/batch-denoise",
                        headers={
                            "x-api-key": self.API_KEY,
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
            return self.convert_output(denoised_array, num_channels = num_channels, output_format = output_format, file_name = file_name)

    sns.set_theme()

    def plot_denoised(
        self,
        data_in, # shape (channels, samples)
        num_channels: int,
        sample_rate: int = 512,
        initial_window_sec: float = 2.0,
    ):
        """
        Create an interactive figure showing the clean and noisy time serives

        :param data_in: array, list, df, or csv
        :param num_channels: number of EEG channels
        :param sample_rate: sampling rate in Hz
        :param initial_window_sec: initial view window width in seconds
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
        denoised_array = self.denoise_batch(
            data_in=data_in,
            num_channels=num_channels,
            sample_rate=sample_rate,
            output_format="array",
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
        self,
        eeg_segment,
    ):
        """
        Find RRMSE for a segment of EEG data
        
        :param eeg_segment: array, list, or df
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
                f"{self.base_url}/denoise-rrmse",
                headers={
                    "x-api-key": self.API_KEY,
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

    def lsl_denoise(
        self,
        stream_duration = 0,
        num_channels = 4,
        sample_rate = 512,
        output_format = "array",
        file_name = "denoised_lsl.csv"
    ):
        """
        Use LSL to stream live data from eeg device into denoising endpoint
        
        :param stream_duration: How long the stream lasts in seconds, 0 means indefinite
        :num_channels: how many channels in the data
        :sample_rate: how many data points per second the eeg device outputs, for optimal results match the batch size of 512
        :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
        :param file_name: Used if output_format='csv'.
        """
        
        batch_size = 512
        buffer_list = []
        denoised_chunks = []
        print("Resolving LSL stream...")
        streams = resolve_streams()
        inlet = StreamInlet(streams[0])
        period = math.ceil(batch_size / sample_rate)
        print(f"period is: {period}")
        print("Starting LSL stream. Press Ctrl+C to stop (if stream_duration=0).")
        initial_time = time.time()
        try:
            while True:
                start_time = time.time()
                if stream_duration > 0:
                    if (time.time() - initial_time) >= stream_duration:
                        print(f"Reached {stream_duration} seconds. Stopping stream.")
                        break
                        
                while (time.time() - start_time) < period:
                    chunk, timestamps = inlet.pull_chunk(timeout=0.2)
                    if timestamps:
                        buffer_list.extend(chunk)
                        
                if len(buffer_list) >= batch_size:
                    data_512 = buffer_list[:batch_size]
                    buffer_list = buffer_list[batch_size:]
                    
                    data_in = np.array(data_512, dtype=np.float32).T  # shape => (4,512)
                    print(f"Collected 512 samples => shape {np.shape(data_in)}. Ready to process.")
                    
                    # pass into denoise_batch
                    denoised_data = self.denoise_batch(
                        data_in=data_in,
                        num_channels = num_channels,
                        sample_rate=512,
                        output_format = "array"
                    )
                    print("denoised data")
                    print(denoised_data)
                    
                    denoised_chunks.append(denoised_data)
                
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Stopping stream...")
        
        if len(denoised_chunks) == 0:
            print("No data was collected or denoised.")
            final_array = np.zeros((num_channels, 0), dtype=np.float32)
        else:
            final_array = np.concatenate(denoised_chunks, axis=1)  # => shape (num_channels, 512*N)

        print("Final shape:", final_array.shape)

        # Convert to requested format
        return self.convert_output(final_array, num_channels = num_channels, output_format=output_format, file_name = file_name)
