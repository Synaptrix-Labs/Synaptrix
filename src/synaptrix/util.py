import time
import math
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import seaborn as sns
from pylsl import StreamInlet, resolve_streams
from scipy import signal
import zstandard as zstd
import pickle

class SynaptrixClient:
    def __init__(self, API_KEY: str, base_url: str = "https://neurodiffusionapi-apim.azure-api.net"):
        self.API_KEY = API_KEY
        self.base_url = base_url

    def apply_notch_filter(self, data, fs, notch_freqs=[50, 60]):
        filtered_data = data.copy()
        
        q_values = [30] * len(notch_freqs)
        for freq, q in zip(notch_freqs, q_values):
            b, a = signal.iirnotch(freq, q, fs)
            # Apply the filter to each channel
            for i in range(data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
    
    def apply_bandpass_filter(self, data, fs, low_freq=1, high_freq=100):
        
        filtered_data = np.zeros_like(data)
        
        nyq = 0.5 * fs
        low = low_freq / nyq
        high = high_freq / nyq
        
        b, a = signal.butter(4, [low, high], btype='band')        
        # Apply the filter to each channel
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
        
    def filter_data(self, data, fs, notch_freqs=[50,60], low_freq=1, high_freq=100):
        filtered_data = self.apply_notch_filter(data, fs, notch_freqs)
        filtered_data = self.apply_bandpass_filter(filtered_data, fs, low_freq, high_freq)
        
        return filtered_data
        
    def reshape_data(self, data, normalize=True, data_columns=None, skip_rows=0, datetime_column=None):
        """
        Internal helper to reshape input data into a nested list, normalize, 
        and extract a datetime column if provided.
        
        """

        # Load/convert data into a list of rows
        if isinstance(data, str):
            df = pd.read_csv(data, skiprows=skip_rows)
            
        elif isinstance(data, pd.DataFrame):
            df = data.iloc[skip_rows:] if skip_rows else data
            
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data.T)
            if skip_rows:
                df = df.iloc[skip_rows:]
                
        elif isinstance(data, list):
            data_rotated = list(map(list, zip(*data)))
            df = pd.DataFrame(data_rotated[skip_rows:])
            
        else:
            raise ValueError("Unsupported data type. Use a numpy array, pandas DataFrame, list of lists, or CSV file path.")

        # Determine columns to process
        if data_columns is None:
            data_columns = list(range(df.shape[1]))

        # Remove datetime column from numeric processing, if provided.
        datetime_data = None
        if datetime_column is not None:
            datetime_data = df.iloc[:, datetime_column].values
            if datetime_column in data_columns:
                data_columns.remove(datetime_column)

        # Select the numeric data and convert to a NumPy array (float conversion)
        numeric_data = df.iloc[:, data_columns].to_numpy(dtype=float)

        # Apply vectorized normalization if requested
        if normalize:
            means = np.mean(numeric_data, axis=0)
            stds = np.std(numeric_data, axis=0)
            # Avoid division by zero
            stds[stds == 0] = 1
            numeric_data = (numeric_data - means) / stds

        # Return with shape (num_points, num_channels)
        return numeric_data.T, datetime_data
    
    def convert_output(
        self,
        data: np.ndarray,
        num_channels: int = 1,
        datetime = None, 
        output_format: str = "array", 
        file_name: str = "denoised.csv"
    ):
        """
        Internal helper function to convert a NumPy array `data` 
        to the user-requested format: array, list, dataframe, or csv.
        
        - `data` is shape (channels, samples).
        - `num_channels` is equal to number of channels user wanted to denoise
        - `datetime` if exists is equal to the column containing datetime data
        - `output_format` can be "array", "list", "df", or "csv".
        - `file_name` can be used if you want to save CSV to disk. 
        """
        
        # Array output
        if output_format.lower() == "array":
            return data 
        
        # List output
        elif output_format.lower() == "list":
            return data.tolist()
        
        # DF and CSV output
        elif output_format.lower() in ["df", "csv"]:
            # Create channel names and transpose the data so that each row is a sample.
            columns = [f"channel_{i+1}" for i in range(num_channels)]
            data_T = data.T
            df = pd.DataFrame(data_T, columns=columns)
            
            # If datetime is provided, insert it as the first column.
            if datetime is not None:
                df.insert(0, "datetime", datetime)
            
            if output_format.lower() == "df":
                return df

            elif output_format.lower() == "csv":
                df.to_csv(file_name, index=False, header=True)
                return file_name

    def compress(self, eeg_array):
        """Internal helper function to compress an EEG NumPy array using Zstandard and return the corresponding byte stream"""
        compressor = zstd.ZstdCompressor(level=3)
        eeg_bytestream = pickle.dumps(eeg_array)
        compressed_eeg_bytestream = compressor.compress(eeg_bytestream)

        return compressed_eeg_bytestream

    def decompress(self, compressed_eeg_bytestream):
        """Internal helper function to decompress a byte stream using Zstandard and return the corresponding EEG NumPy array"""
        decompressor = zstd.ZstdDecompressor()
        eeg_bytestream = decompressor.decompress(compressed_eeg_bytestream)
        eeg_array = pickle.loads(eeg_bytestream)

        return eeg_array
    
    def calculate_SPPs_used(self, eeg_array):
        SPPs_per_channel, num_channels = eeg_array.shape
        return num_channels * SPPs_per_channel

    def denoise_batch(
        self,
        data_in,
        normalize: bool = True,
        data_columns = None,
        skip_rows: int = 0,
        datetime_column = None,
        filter: bool = True,
        sample_rate: int = 512,
        notch_freqs: list = [60],
        low_freq: int = 1,
        high_freq: int = 100,
        output_format: str = "array",
        file_name: str = "denoised_batch.csv",
    ):
        """
        Denoise a multi channel and time series as long as you want.
        
        :param data_in: array, list, df, or csv
        :param normalize: bool, default True. If True, each channel will be normalized (z-score) before chunking.
        :param data_columns: an array of the indices of the columns that the user wants to denoise
        :param skip_rows: an integer equaling to the number of rows off the top of the df or csv the user wants to skip
        :param datetime_column: an integer equaling to the index of the column that contains datetime data, default None
        :param filter: set this parameter to False if your input data is already filtered, default is True
        :param sample_rate: an integer equaling to the sample rate of your data
        :param notch_freqs: a list of integers equaling to which hz you want to apply a notch filter
        :param low_freq: an integer equaling to the lower bound of the bandpass filter
        :param high_freq: an integer equaling to the higher bound of teh bandpass filter
        :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
        :param file_name: Used if output_format='csv'.
        """

        if normalize:
            if filter:
                print("Filtering data...")

                reshaped_data_in, datetime_data = self.reshape_data(data=data_in, normalize=False, data_columns=data_columns, skip_rows=skip_rows, datetime_column=datetime_column)
                data_in_array = self.filter_data(reshaped_data_in, fs=sample_rate, notch_freqs=notch_freqs, low_freq=low_freq, high_freq=high_freq)
                filtered_data_in = pd.DataFrame(data_in_array.T)
                data_in_array, _ = self.reshape_data(data=filtered_data_in, normalize=True)
            else:
                data_in_array, datetime_data = self.reshape_data(data=data_in, normalize=normalize, data_columns=data_columns, skip_rows=skip_rows, datetime_column=datetime_column)
        else:
            print("Normalization is off.")

            if filter:
                print("Filtering data...")

                reshaped_data_in, datetime_data = self.reshape_data(data=data_in, normalize=normalize, data_columns=data_columns, skip_rows=skip_rows, datetime_column=datetime_column)
                data_in_array = self.filter_data(reshaped_data_in, fs=sample_rate, notch_freqs=notch_freqs, low_freq=low_freq, high_freq=high_freq)
                filtered_data_in = pd.DataFrame(data_in_array.T)
                data_in_array, _ = self.reshape_data(data=filtered_data_in, normalize=normalize)
            else:
                data_in_array, datetime_data = self.reshape_data(data=data_in, normalize=normalize, data_columns=data_columns, skip_rows=skip_rows, datetime_column=datetime_column)
        
        # Compress nested_data before sending through API endpoint
        compressed_eeg_bytestream = self.compress(data_in_array)

        try:
            # Make a single API call with the compressed bytestream
            print("Denoising data...")
            
            response = requests.post(
                f"{self.base_url}/batch-denoise",
                headers={
                    "x-api-key": self.API_KEY,
                    "Content-Type": "application/octet-stream"
                },
                data=compressed_eeg_bytestream
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            # Add more detailed error info
            try:
                error_details = response.json()
                error_message = f"Request failed: {e}. Details: {error_details}"
            except:
                error_message = f"Request failed: {e}"
            raise RuntimeError(error_message)

        # Process the response
        denoised_eeg_bytestream = response.content
        
        # Decompress the output from API call
        denoised_array = self.decompress(denoised_eeg_bytestream)

        if denoised_array is None:
            raise ValueError("Received empty denoised data from the API.")
        
        spp_consumed = self.calculate_SPPs_used(reshaped_data_in)
        print(f"Denoising completed - this operation consumed {spp_consumed} SPP's.")

        return self.convert_output(denoised_array, num_channels = denoised_array.shape[0], datetime = datetime_data, output_format = output_format, file_name = file_name)

    sns.set_theme()

    def plot_denoised(
        self,
        data_in, # shape (channels, samples)
        normalize: bool = True,
        data_columns = None,
        skip_rows: int = 0,
        filter = True,
        sample_rate: int = 512,
        notch_freqs: list = [60],
        low_freq: int = 1,
        high_freq: int = 100,
        initial_window_sec: float = 2.0,
    ):
        """
        Create an interactive figure showing the clean and noisy time serives

        :param data_in: array, list, df, or csv
        :param normalize: bool, default True. If True, each channel will be normalized (z-score) before chunking.
        :param data_columns: an array of the indices of the columns that the user wants to denoise
        :param skip_rows: an integer equaling to the number of rows off the top of the df or csv the user wants to skip
        :param filter: set this parameter to False if your input data is already filtered, default is True
        :param sample_rate: an integer equaling to the sample rate of your data
        :param notch_freqs: a list of integers equaling to which hz you want to apply a notch filter
        :param low_freq: an integer equaling to the lower bound of the bandpass filter
        :param high_freq: an integer equaling to the higher bound of teh bandpass filter
        :param initial_window_sec: initial view window width in seconds
        """
        
        if normalize:
            if filter:
                data_in_list, _ = self.reshape_data(
                    data = data_in,
                    normalize=False,
                    data_columns=data_columns,
                    skip_rows=skip_rows,
                )
                prefilter_data_in_array = np.array(data_in_list)
                data_in_array = self.filter_data(prefilter_data_in_array, fs=sample_rate, notch_freqs=notch_freqs, low_freq=low_freq, high_freq=high_freq)
                data_in_array, _ = self.reshape_data(data=data_in_array, normalize=True)
                                
            else:
                data_in_list, _ = self.reshape_data(
                    data = data_in,
                    normalize=normalize,
                    data_columns=data_columns,
                    skip_rows=skip_rows,
                )
                data_in_array = np.array(data_in_list)
            
        else:
            print("Normalization is off.")
            if filter:
                data_in_list, _ = self.reshape_data(
                    data = data_in,
                    normalize=normalize,
                    data_columns=data_columns,
                    skip_rows=skip_rows,
                )
                prefilter_data_in_array = np.array(data_in_list)
                data_in_array = self.filter_data(prefilter_data_in_array, fs=sample_rate, notch_freqs=notch_freqs, low_freq=low_freq, high_freq=high_freq)
                data_in_array, _ = self.reshape_data(data=data_in_array, normalize=False)
                                
            else:
                data_in_list, _ = self.reshape_data(
                    data = data_in,
                    normalize=normalize,
                    data_columns=data_columns,
                    skip_rows=skip_rows,
                )
                data_in_array = np.array(data_in_list)
                
        denoised_array = self.denoise_batch(
            data_in=data_in,
            normalize=normalize,
            data_columns=data_columns,
            skip_rows=skip_rows,
            filter = filter,
            sample_rate = sample_rate,
            notch_freqs = notch_freqs,
            low_freq = low_freq,
            high_freq = high_freq,
            output_format="array",
        )

        channels, total_samples = denoised_array.shape
        
        # Create subplot for each channel
        fig, axes = plt.subplots(nrows=channels, ncols=1, sharex=True, figsize=(10, 6))
        if channels == 1:
            axes = [axes]

        fig.suptitle("NeuroDiffusion (Denoised & Noisy)", fontsize=14)
        time = np.arange(total_samples) / sample_rate

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

        # Toggle noisy lines on/off
        show_noisy = False

        # Update function to redraw lines based on current window
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

        # Button callbacks (Left/Right):
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

        # Update window width
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

        # Show/hide noisy lines
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
        return print(f"The RRMSE of this segment is {RRMSE}.")

    def lsl_denoise(
        self,
        normalize: bool=True,
        stream_duration = 0,
        num_channels = 4,
        sample_rate = 512,
        filter = True,
        notch_freqs=[50,60],
        low_freq: int = 1,
        high_freq: int = 100,
        output_format = "array",
        file_name = "denoised_lsl.csv"
    ):
        """
        Use LSL to stream live data from eeg device into denoising endpoint
        
        :param normalize: bool, default True. If True, each channel will be normalized (z-score) before chunking.
        :param stream_duration: How long the stream lasts in seconds, 0 means indefinite
        :num_channels: how many channels in the data
        :sample_rate: how many data points per second the eeg device outputs, for optimal results match the batch size of 512
        :param filter: set this parameter to False if your input data is already filtered, default is True
        :param sample_rate: an integer equaling to the sample rate of your data
        :param notch_freqs: a list of integers equaling to which hz you want to apply a notch filter
        :param low_freq: an integer equaling to the lower bound of the bandpass filter
        :param high_freq: an integer equaling to the higher bound of teh bandpass filter
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
                    
                    data_in = np.array(data_512, dtype=np.float32).T
                    print(f"Collected 512 samples => shape {np.shape(data_in)}. Ready to process.")
                    
                    # Pass into denoise_batch
                    denoised_data = self.denoise_batch(
                        data_in=data_in,
                        normalize=normalize,
                        filter=filter,
                        sample_rate=sample_rate,
                        notch_freqs=notch_freqs,
                        low_freq=low_freq,
                        high_freq=high_freq,
                        output_format = "array"
                    )
                    print("Denoised data:")
                    print(denoised_data)
                    
                    denoised_chunks.append(denoised_data)
                
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Stopping stream...")
        
        if len(denoised_chunks) == 0:
            print("No data was collected or denoised.")
            final_array = np.zeros((num_channels, 0), dtype=np.float32)
        else:
            final_array = np.concatenate(denoised_chunks, axis=1)

        print("Final shape:", final_array.shape)
        total_spp = int(final_array.shape[0]* (final_array.shape[1]/512)*512)

        # Convert to requested format
        print(f"This LSL stream consumed {total_spp} SPP's")
        return self.convert_output(final_array, num_channels = num_channels, output_format=output_format, file_name = file_name)


