# Synaptrix

A library for live and offline denoising of multi-channel EEG data powered by auto-encoders, created by Synaptrix Labs Inc. This README covers how to prepare **LabStreamingLayer (LSL)** for your platform, and how to install and initialize the `SynaptrixClient`.



## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [Install LSL on macOS](#install-lsl-on-macos)
  - [Install LSL on Windows](#install-lsl-on-windows)
  - [Install LSL on Linux](#install-lsl-on-linux)
- [Installation](#installation)
- [Usage](#usage)
  - [Initialize `SynaptrixClient`](#initialize-synaptrixclient)
- [License](#license)



## Overview

**Synaptrix** provides a convenient Python API for EEG denoising using your own model or pre-trained models. It also integrates with **LabStreamingLayer (LSL)** for real-time data acquisition. To use LSL functionality, you must first install the **native** LSL libraries on your system (see [Prerequisites](#prerequisites)).



## Prerequisites

### Install LSL on macOS

1. Ensure you have [Homebrew](https://brew.sh/) installed.
2. Run:
    ```bash
    brew install labstreaminglayer/tap/lsl
    ```
This installs the native LSL libraries that pylsl depends on.

### Install LSL on Windows
1.	Visit the official [LSL Windows Installation Docs](https://github.com/sccn/labstreaminglayer)
2.	Download the appropriate installer/zip.
3.	Follow instructions to install the .dlls so that pylsl can detect them.

### Install LSL on Linux
1.	Ubuntu/Debian (example):
    ```bash
    sudo apt-get update
    sudo apt-get install cmake build-essential
    git clone https://github.com/sccn/labstreaminglayer.git
    cd labstreaminglayer/LSL
    # Then follow build instructions from the official docs
    ```



## Installation

After installing the native LSL libraries for your platform, you can install Synaptrix:
```bash
    pip install synaptrix
```



## Usage

### Initialize SynaptrixClient

```python
    from synaptrix import SynaptrixClient
    import pandas as pd

    # Initialize the client
    client = SynaptrixClient(
        API_KEY="YourAPIKey"
    )
```

After initializing the client, you can then access all the functions of synaptrix.

Our model works best with filtered and normalized data. If your data is not preprocessed don't worry! The library will apply a notch filter at 50 and 60hz (powerline), a bandpass filter from 1-100hz, and normalize your data by default. By setting the parameter "Normalize" to False, we will return the denoised data in your original scaling.

Here is an example of how you can denoise a csv file called data.csv. Let's say the columns you want to denoise has indices [1,2,5,7], you want to skip the first 2 rows of data because they contain header data, the column with index 0 contains datetime data, your data is at 512hz, and you want to output a df:
    
```python
    data_in = pd.read_csv("data.csv")
    
    denoised = client.denoise_batch(
        data_in,
        data_columns=[1,2,5,7],
        skip_rows=2,
        datetime_column=0,
        sample_rate=512,
        output_format="df") 
    
    print("Denoised Data: ", denoised)
```
Now let's say you have a dataset that is already filtered. Set the filter parameter to False first. Now let's say you want to denoise columns with indices [4,6,7], you want to skip no rows of data, the datetime column has index 3, and you want to output a csv called "denoised_data.csv":
    
```python
    data_in = pd.read_csv("data.csv")
    
    client.denoise_batch(
        data_in,
        normalize=False,
        data_columns=[4,6,7],
        filter=False,
        skip_rows=0,
        datetime_column=3,
        sample_rate=512,
        output_format="csv",
        file_name="denoised_data.csv") 
```
You can also customize the placement of the notch and bandpass filters. Just keep in mind that any filter must be less than half of your sampling rate. If the data has a sampling rate of 200hz, the highest filter you can apply is 99hz:
    
```python
    data_in = pd.read_csv("data.csv")
    
    client.denoise_batch(
        data_in,
        data_columns=[4,6,7],
        datetime_column=3,
        sample_rate=512,
        notch_freqs=[45,70],
        low_freq=15,
        high_freq=75,
        output_format="csv",
        file_name="denoised_data.csv")

```
Here is an example of how you can generate a plot of the denoised data:
    
```python
    data_in = pd.read_csv("data.csv)
    client.plot_denoised(
        data_in,
        data_columns=[4,6,7], #remove this parameter to plot all columns
        datetime_column=3,
        sample_rate=512,
        initial_window_sec=2)
    
    # initial_window_sec dictates how wide is the sliding viewing window
```
Here is an example of how to stream data through lsl into synaptrix and output denoised data:

```python
    lsl_output = client.lsl_denoise(
        stream_duration = 10, # in seconds, change parameter to 0 for indefinite streaming
        num_channels = 4,
        sample_rate = 512, # adjust to match sampling rate of your device
        output_format = "csv",
        file_name = "denoised_lsl.csv") 
        
    # at the conclusion of the stream, all denoised data will be saved to this file
```
---

## License
This project is licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](https://github.com/Synaptrix-Labs/Synaptrix/blob/main/LICENSE) file for details.
