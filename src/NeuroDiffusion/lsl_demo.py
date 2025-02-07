import time
import numpy as np
from pylsl import StreamInlet, resolve_streams
from util import denoise_batch

def main(api_key="rinklybrain"):
    # Resolve the LSL stream from the Ganglion board
    print("Resolving LSL stream...")
    streams = resolve_streams()
    inlet = StreamInlet(streams[0])

    # ganglion info
    num_channels = 4
    sample_rate = 200 
    buffer_list = [] 
    batch_size = 512
    interval_sec = 3.0
    
    print(f"Collecting data at 200 Hz, grabbing 512-sample batches (~2.56s) every 3s...")

    while True:
        start_time = time.time()
        # Run for ~3 seconds, accumulating data from LSL
        while time.time() - start_time < interval_sec:
            chunk, timestamps = inlet.pull_chunk(timeout=0.5)
            if timestamps:
                buffer_list.extend(chunk)
        
        # After 3 seconds, see if we have at least 512 samples
        if len(buffer_list) >= batch_size:
            data_512 = buffer_list[:batch_size]
            buffer_list = buffer_list[batch_size:]
            
            data_4x512 = np.array(data_512, dtype=np.float32).T  # shape => (4,512)
            
            # pass into denoise_batch
            denoised_4x512 = denoise_batch(
                data_in=data_4x512,
                api_key=api_key,
                num_channels=4,
                sample_rate=512,
                output_format="array"
            )
            
            print(f"Collected 512 samples => shape {np.shape(data_4x512)}. Ready to process.")
            print("denoised data")
            print(denoised_4x512)
            print(np.shape(denoised_4x512))
        else:
            print(f"Only collected {len(buffer_list)} samples in 3s, not enough for 512, skipping.")


if __name__ == "__main__":
    main()