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


    def reshape_data(self, data, data_columns=None, skip_rows=0, datetime_column=None):
        """
        Reshape input data into a nested list and extract a datetime column if provided.
        
        Parameters
        ----------
        data : np.ndarray, pd.DataFrame, list of lists, or str (path to CSV file)
            The input EEG data.
        data_columns : list of int, optional
            Column indices to process. If None, all columns are processed.
        skip_rows : int, optional
            Number of rows to skip from the top of the data (default is 0).
        datetime_column : int, optional
            Column index that contains datetime data. If provided, that column is extracted 
            separately.
            
        Returns
        -------
        reshaped_data : list of lists
            A nested list with dimensions (len(data_columns), number of data points per column).
        datetime_data : list or None
            A list of datetime values if datetime_column is provided; otherwise, None.
        """
        # Load/convert data into a list of rows
        if isinstance(data, str):
            df = pd.read_csv(data, skiprows=skip_rows)
            data_list = df.values.tolist()
        elif isinstance(data, pd.DataFrame):
            if skip_rows:
                data = data.iloc[skip_rows:, :]
            data_list = data.values.tolist()
        elif isinstance(data, np.ndarray):
            if skip_rows:
                data = data[skip_rows:, :]
            data_list = data.tolist()
        elif isinstance(data, list):
            # Assume it's already a list of lists (each inner list is a row)
            data_list = data[skip_rows:]
        else:
            raise ValueError("Unsupported data type. Please use a numpy array, pandas DataFrame, list of lists, or CSV file path.")

        if not data_list:
            raise ValueError("No data available after applying skip_rows.")

        # If no columns are specified, process all columns
        if data_columns is None:
            data_columns = list(range(len(data_list[0])))

        datetime_data = None
        if datetime_column is not None:
            datetime_data = [row[datetime_column] for row in data_list]

        # Build the nested list for the selected data columns
        reshaped_data = []
        for col in data_columns:
            col_data = [row[col] for row in data_list]
            reshaped_data.append(col_data)

        return reshaped_data, datetime_data
    
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
        
        # array
        if output_format.lower() == "array":
            return data 
        
        # list
        elif output_format.lower() == "list":
            return data.tolist()
        
        # df and csv
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
        data_columns = None,
        skip_rows: int = 0,
        datetime_column = None,
        output_format: str = "array",
        file_name: str = "denoised_batch.csv",
    ):
        """
        Denoise a multi channel and time series as long as you want.
        
        :param data_in: array, list, df, or csv
        :param data_columns: an array of the indices of the columns that the user wants to denoise
        :param skip_rows: an integer equaling to the number of rows off the top of the df or csv the user wants to skip
        :param datetime_column: an integer equaling to the index of the column that contains datetime data, default None
        :param output_format: Desired output format: 'array', 'list', 'df', or 'csv'.
        :param file_name: Used if output_format='csv'.
        """
        window_size = 512
        windows = []
        data_in_list, datetime = self.reshape_data(data_in, data_columns, skip_rows, datetime_column)
        
        
        #will store each denoised channel into this array
        denoised_array = []
        
        #check if each channel is a multiple of sampling rate
        data_in_length = np.shape(data_in_list)[1]
        for chan in data_in_list:
            windows = []
            if data_in_length % window_size != 0:
                caboose = data_in_length % window_size
                chan = chan[:-caboose]
            
            #chunk data from each channel by the sampling rate and append to windows
            num_windows = int(data_in_length / window_size)
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
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
        if datetime:
            datetime = datetime[:-caboose]
        return self.convert_output(denoised_array, num_channels = np.shape(data_in_list)[0], datetime = datetime, output_format = output_format, file_name = file_name)

    sns.set_theme()

    def plot_denoised(
        self,
        data_in, # shape (channels, samples)
        data_columns = None,
        skip_rows: int = 0,
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
        
        #Reshape inputted noisy data
        data_in_list, _ = self.reshape_data(
            data = data_in,
            data_columns=data_columns,
            skip_rows=skip_rows,
        )
        data_in_array = np.array(data_in_list)
        #Call denoise_batch
        denoised_array = self.denoise_batch(
            data_in=data_in,
            data_columns=data_columns,
            skip_rows=skip_rows,
            output_format="array",
        )
        channels, total_samples = denoised_array.shape
        

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


