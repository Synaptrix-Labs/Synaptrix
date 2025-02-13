# Synaptrix

A library for live and offline denoising of multi-channel EEG data powered by auto-encoders, created by Synaptrix Labs Inc. This README covers how to prepare **LabStreamingLayer (LSL)** for your platform, and how to install and initialize the `SynaptrixClient`.

---

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

---

## Overview

**Synaptrix** provides a convenient Python API for EEG denoising using your own model or pre-trained models. It also integrates with **LabStreamingLayer (LSL)** for real-time data acquisition. To use LSL functionality, you must first install the **native** LSL libraries on your system (see [Prerequisites](#prerequisites)).

---

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

---

## Installation

After installing the native LSL libraries for your platform, you can install Synaptrix:
    ```bash
    pip install synaptrix
    ```

---

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

Here is an example of how you can denoise a csv file called data.csv containing 4 channels of eeg data and output as a df:
    
    ```python
    data_in = pd.read_csv("data.csv")
    denoised = client.denoise_batch(data_in, num_channels=4, output_format="df") # output_format can be adjusted to "array", "list", "df, or "csv"
    print("Denoised Data: ", denoised)
    ```
Here is an example of how you can generate a plot of the denoised data:
    
    ```python
    data_in = pd.read_csv("data.csv)
    client.plot_denoised(data_in, num_channels=4, initial_window_sec=1) # initial_window_sec dictates how wide is the sliding window for viewing the plot
    ```
Here is an exmaple of how to stream data through lsl into synaptrix and output denoised data:

    ```python
    lsl_output = client.lsl_denoise(
        stream_duration = 10, # in seconds, change parameter to 0 for indefinite streaming
        num_channels = 4,
        sample_rate = 512, # adjust to match sampling rate of your device
        output_format = "csv",
        file_name = "lsl_test.csv" # at the conclusion of the stream, all denoised data will be saved to this file
    )
    ```
---

## License
This project is licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](https://github.com/Synaptrix-Labs/Synaptrix/blob/main/LICENSE) file for details.