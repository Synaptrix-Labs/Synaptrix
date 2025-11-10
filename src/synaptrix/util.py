import time
import math
from collections import deque
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
        self._compressor = zstd.ZstdCompressor(level=3)
        self._decompressor = zstd.ZstdDecompressor()
        self._session = requests.Session()
        self._session.headers.update({"x-api-key": self.API_KEY})

    def apply_notch_filter(self, data, fs, notch_freqs=[50, 60]):
        """
        Apply sequential notch filters to every channel at once (axis=1).
        """
        filtered = np.asarray(data).copy()
        q = 30
        for freq in notch_freqs:
            b, a = signal.iirnotch(freq, q, fs)
            filtered = signal.filtfilt(b, a, filtered, axis=1)

        return filtered
    
    def apply_bandpass_filter(self, data, fs, low_freq=0.5, high_freq=40):
        """
        Apply a Butterworth band-pass to all channels using SOS form for stability.
        """
        data_arr = np.asarray(data, dtype=float)

        nyq = 0.5 * fs
        low = low_freq / nyq
        high = high_freq / nyq

        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, data_arr, axis=1)
        
    def filter_data(self, data, fs, notch_freqs=[50,60], low_freq=0.5, high_freq=40):
        filtered_data = self.apply_notch_filter(data, fs, notch_freqs)
        filtered_data = self.apply_bandpass_filter(filtered_data, fs, low_freq, high_freq)
        
        return filtered_data
        
    def reshape_data(self, data, normalize=True, data_columns=None, skip_rows=0, datetime_column=None):
        """
        Internal helper to reshape input data into a nested list, normalize, 
        and extract a datetime column if provided.
        
        """

        # Load/convert the data so rows = samples and columns = channels
        if isinstance(data, str):
            df = pd.read_csv(data, skiprows=skip_rows)
            base_array = df.to_numpy()
        elif isinstance(data, pd.DataFrame):
            df = data.iloc[skip_rows:] if skip_rows else data
            base_array = df.to_numpy()
        else:
            base_array = np.asarray(data)
            if base_array.ndim == 1:
                base_array = base_array[:, None]
            # Incoming arrays/lists are assumed channel-first; match historical behavior.
            base_array = base_array.T
            if skip_rows:
                base_array = base_array[skip_rows:]

        if base_array.size == 0:
            raise ValueError("Input data is empty after processing.")

        n_cols = base_array.shape[1]
        numeric_cols = list(range(n_cols)) if data_columns is None else list(data_columns)

        datetime_data = None
        if datetime_column is not None:
            datetime_data = base_array[:, datetime_column]
            numeric_cols = [idx for idx in numeric_cols if idx != datetime_column]

        if not numeric_cols:
            raise ValueError("No numeric columns available for processing.")

        numeric_data = base_array[:, numeric_cols].astype(float, copy=False)

        means = None
        stds = None
        if normalize:
            means = np.mean(numeric_data, axis=0)
            stds = np.std(numeric_data, axis=0)
            stds[stds == 0] = 1
            numeric_data = (numeric_data - means) / stds

        return numeric_data.T, datetime_data, means, stds
    
    def strangeify(self, data, means, stds):
        """
        Apply un-normalization to return data to its original scale.
        
        :param data: Normalized data with shape (channels, samples)
        :param means: Mean values used during normalization
        :param stds: Standard deviation values used during normalization
        :return: Un-normalized data in the original scale
        """
        if means is None or stds is None:
            return data
            
        # Transpose to have channels as columns
        data_T = data.T
        un_normalized = data_T * stds + means
        return un_normalized.T
    
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
        eeg_bytestream = pickle.dumps(eeg_array)
        compressed_eeg_bytestream = self._compressor.compress(eeg_bytestream)

        return compressed_eeg_bytestream

    def decompress(self, compressed_eeg_bytestream):
        """Internal helper function to decompress a byte stream using Zstandard and return the corresponding EEG NumPy array"""
        eeg_bytestream = self._decompressor.decompress(compressed_eeg_bytestream)
        eeg_array = pickle.loads(eeg_bytestream)

        return eeg_array
    
    def calculate_SPPs_used(self, eeg_array):
        SPPs_per_channel, num_channels = eeg_array.shape
        return num_channels * SPPs_per_channel

    def _normalize_channels(self, channel_first_array):
        """
        Normalize each channel of a (channels, samples) array.
        """
        means = np.mean(channel_first_array, axis=1, keepdims=True)
        stds = np.std(channel_first_array, axis=1, keepdims=True)
        stds[stds == 0] = 1
        normalized = (channel_first_array - means) / stds
        return normalized, means.ravel(), stds.ravel()

    def mise_en_place(
        self,
        data_in,
        data_columns,
        skip_rows,
        datetime_column,
        filter,
        sample_rate,
        notch_freqs,
        low_freq,
        high_freq,
    ):
        """
        Shared preprocessing pipeline used by denoise_batch and plot_denoised.
        """
        channel_array, datetime_data, _, _ = self.reshape_data(
            data=data_in,
            normalize=False,
            data_columns=data_columns,
            skip_rows=skip_rows,
            datetime_column=datetime_column,
        )

        channel_array = np.asarray(channel_array, dtype=float)

        if filter:
            channel_array = self.filter_data(
                channel_array,
                fs=sample_rate,
                notch_freqs=notch_freqs,
                low_freq=low_freq,
                high_freq=high_freq,
            )

        normalized_array, means, stds = self._normalize_channels(channel_array)

        return normalized_array, channel_array, datetime_data, means, stds

    def _run_batch_denoise(self, normalized_array):
        compressed_eeg_bytestream = self.compress(normalized_array)

        try:
            print("Denoising data...")
            response = self._session.post(
                f"{self.base_url}/batch-denoise",
                headers={"Content-Type": "application/octet-stream"},
                data=compressed_eeg_bytestream,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            try:
                error_details = response.json()
                error_message = f"Request failed: {e}. Details: {error_details}"
            except Exception:
                error_message = f"Request failed: {e}"
            raise RuntimeError(error_message)

        denoised_array = self.decompress(response.content)
        if denoised_array is None:
            raise ValueError("Received empty denoised data from the API.")

        spp_consumed = self.calculate_SPPs_used(denoised_array)
        print(f"Denoising completed - this operation consumed {spp_consumed} SPP's.")

        return denoised_array

    def denoise_batch(
        self,
        data_in,
        normalize: bool = False,
        data_columns = None,
        skip_rows: int = 0,
        datetime_column = None,
        filter: bool = True,
        sample_rate: int = 512,
        notch_freqs: list = [60],
        low_freq: int = 0.5,
        high_freq: int = 40,
        output_format: str = "array",
        file_name: str = "denoised_batch.csv",
    ):
        """
        Denoise a multi channel and time series as long as you want.
        
        :param data_in: array, list, df, or csv
        :param normalize: bool, default False. If True, the output will be in normalized space.
           If False, the output will be un-normalized back to the original scale.
        :param data_columns: an array of the indices of the columns that the user wants to denoise
        :param skip_rows: an integer equaling to the number of rows off the top of the df or csv the user wants to skip
        :param datetime_column: an integer equaling to the index of the column that contains datetime data, default None
        :param filter: set this parameter to False if your input data is already filtered, default is True
        :param sample_rate: an integer equaling to the sample rate of your data
        :param notch_freqs: a list of integers equaling to which hz you want to apply a notch filter
        :param low_freq: an integer equaling to the lower bound of the bandpass filter
        :param high_freq: an integer equaling to the higher bound of the bandpass filter
        :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
        :param file_name: Used if output_format='csv'.
        """

        normalized_array, _, datetime_data, means, stds = self.mise_en_place(
            data_in=data_in,
            data_columns=data_columns,
            skip_rows=skip_rows,
            datetime_column=datetime_column,
            filter=filter,
            sample_rate=sample_rate,
            notch_freqs=notch_freqs,
            low_freq=low_freq,
            high_freq=high_freq,
        )

        denoised_array = self._run_batch_denoise(normalized_array)

        if not normalize:
            denoised_array = self.strangeify(denoised_array, means, stds)

        return self.convert_output(denoised_array, num_channels = denoised_array.shape[0], datetime = datetime_data, output_format = output_format, file_name = file_name)

    sns.set_theme()

    def plot_denoised(
        self,
        data_in, # shape (channels, samples)
        normalize: bool = False,
        data_columns = None,
        skip_rows: int = 0,
        filter = True,
        sample_rate: int = 512,
        notch_freqs: list = [60],
        low_freq: int = 0.5,
        high_freq: int = 40,
        initial_window_sec: float = 2.0,
    ):
        """
        Create an interactive figure showing the clean and noisy time serives

        :param data_in: array, list, df, or csv
        :param normalize: bool, default False. If True, both noisy and denoised data will be plotted in normalized space.
           If False, data will be plotted in the original scale.
        :param data_columns: an array of the indices of the columns that the user wants to denoise
        :param skip_rows: an integer equaling to the number of rows off the top of the df or csv the user wants to skip
        :param filter: set this parameter to False if your input data is already filtered, default is True
        :param sample_rate: an integer equaling to the sample rate of your data
        :param notch_freqs: a list of integers equaling to which hz you want to apply a notch filter
        :param low_freq: an integer equaling to the lower bound of the bandpass filter
        :param high_freq: an integer equaling to the higher bound of the bandpass filter
        :param initial_window_sec: initial view window width in seconds
        """
        
        normalized_array, noisy_array, _, means, stds = self.mise_en_place(
            data_in=data_in,
            data_columns=data_columns,
            skip_rows=skip_rows,
            datetime_column=None,
            filter=filter,
            sample_rate=sample_rate,
            notch_freqs=notch_freqs,
            low_freq=low_freq,
            high_freq=high_freq,
        )

        denoised_array = self._run_batch_denoise(normalized_array)

        if normalize:
            data_in_array = normalized_array
        else:
            data_in_array = noisy_array
            denoised_array = self.strangeify(denoised_array, means, stds)

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

        def _set_channel_ylim(axis, den_segment, noisy_segment):
            data_min = np.min(den_segment)
            data_max = np.max(den_segment)
            if noisy_segment.size:
                data_min = min(data_min, np.min(noisy_segment))
                data_max = max(data_max, np.max(noisy_segment))
            if data_min == data_max:
                data_min -= 1
                data_max += 1
            pad = 0.05 * (data_max - data_min)
            axis.set_ylim(data_min - pad, data_max + pad)

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

            _set_channel_ylim(ax, denoised_array[ch, start_idx:end_idx], data_in_array[ch, start_idx:end_idx])

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

                _set_channel_ylim(
                    axes[ch],
                    denoised_array[ch, start_idx:end_idx],
                    data_in_array[ch, start_idx:end_idx],
                )

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
            response = self._session.post(
                f"{self.base_url}/denoise-rrmse",
                headers={"Content-Type": "application/json"},
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
        normalize: bool=False,
        stream_duration = 0,
        num_channels = None,
        sample_rate = 512,
        filter = True,
        notch_freqs=[50,60],
        low_freq: int = 0.5,
        high_freq: int = 40,
        output_format = "array",
        file_name = "denoised_lsl.csv"
    ):
        """
        Use LSL to stream live data from eeg device into denoising endpoint
        
        :param normalize: bool, default False. If False, the output will be un-normalized back to the original scale.
        :param stream_duration: How long the stream lasts in seconds, 0 means indefinite
        :param num_channels:
            - None: use all channels from the LSL stream.
            - int: use the first `num_channels` channels.
            - list/tuple of int: indices of LSL channels to process (0-based).        
        :param sample_rate: how many data points per second the eeg device outputs, for optimal results match the batch size of 512
        :param filter: set this parameter to False if your input data is already filtered, default is True
        :param sample_rate: an integer equaling to the sample rate of your data
        :param notch_freqs: a list of integers equaling to which hz you want to apply a notch filter
        :param low_freq: an integer equaling to the lower bound of the bandpass filter
        :param high_freq: an integer equaling to the higher bound of the bandpass filter
        :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
        :param file_name: Used if output_format='csv'.
        """
        
        batch_size = sample_rate
        buffer = deque()
        denoised_chunks = []
        
        print("Resolving LSL stream...")
        streams = resolve_streams()
        inlet = StreamInlet(streams[0])
        info = inlet.info()
        stream_n_channels = info.channel_count()


        if num_channels is None:
            channel_indices = list(range(stream_n_channels))
        elif isinstance(num_channels, int):
            channel_indices = list(range(min(num_channels, stream_n_channels)))
        else:
            channel_indices = list(num_channels)

        if len(channel_indices) == 0:
            raise ValueError("No channel indices specified.")
        if max(channel_indices) >= stream_n_channels or min(channel_indices) < 0:
            raise ValueError(
                f"Channel indices {channel_indices} out of range for LSL stream "
                f"with {stream_n_channels} channels."
            )
        print(f"Using channels (0-based indices): {channel_indices}")
        effective_n_channels = len(channel_indices)


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
                    if timestamps and chunk:
                        buffer.extend(chunk)
                        
                if len(buffer) >= batch_size:
                    data_all = [buffer.popleft() for _ in range(batch_size)]
                    data_full = np.array(data_all, dtype=np.float32).T
                    data_in = data_full[channel_indices, :]
                    print(f"Collected {sample_rate} samples => shape {np.shape(data_in)}. Ready to process.")
                    
                    if filter:
                        data_in = self.filter_data(
                            data_in,
                            fs=sample_rate,
                            notch_freqs=notch_freqs,
                            low_freq=low_freq,
                            high_freq=high_freq,
                        )
                    
                    normalized_chunk, means, stds = self._normalize_channels(data_in)
                    denoised_data = self._run_batch_denoise(normalized_chunk)
                    
                    # Un-normalize if requested
                    if not normalize:
                        denoised_data = self.strangeify(denoised_data, means, stds)
                    
                    print("Denoised data:")
                    print(denoised_data)
                    
                    denoised_chunks.append(denoised_data)
                
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Stopping stream...")
        
        if len(denoised_chunks) == 0:
            print("No data was collected or denoised.")
            final_array = np.zeros((effective_n_channels, 0), dtype=np.float32)
        else:
            final_array = np.concatenate(denoised_chunks, axis=1)

        print("Final shape:", final_array.shape)
        total_spp = int(final_array.shape[0]* (final_array.shape[1]/batch_size)*batch_size)

        # Convert to requested format
        print(f"This LSL stream consumed {total_spp} SPP's")
        return self.convert_output(final_array, num_channels = effective_n_channels, output_format=output_format, file_name = file_name)
