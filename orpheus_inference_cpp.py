import os
import sys
import time

# --- Force Correct CUDA_PATH ---
# Define the correct path WITHOUT surrounding quotes
correct_cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
print(f"System Environment CUDA_PATH: {os.environ.get('CUDA_PATH')}") # Log current value (if any)
print(f"Setting CUDA_PATH for this script run to: {correct_cuda_path}")
os.environ['CUDA_PATH'] = correct_cuda_path
# Add the corresponding bin directory to PATH as well for good measure
cuda_bin_path = os.path.join(correct_cuda_path, 'bin')
if cuda_bin_path not in os.environ['PATH']:
    print(f"Adding {cuda_bin_path} to PATH for this script run.")
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']
# --- END: Force Correct CUDA_PATH ---


# --- Now import libraries that depend on CUDA ---
from scipy.io.wavfile import write
from orpheus_cpp import OrpheusCpp
import numpy as np

# --- Hardcoded Parameters ---
TEST_TEXT = "Okay, <chuckle> we've got a situation here. A woman in the Maine woods... with her dog. Classic setup. But the plot twist? It's not about the dog. It's about... a stick. And she's treating this stick like it's the Hope Diamond. 'I'm currently in the woods of Maine...' as if this warrants a news report. The dog, meanwhile, is operating at peak canine indifference. <laugh> You can practically see it thinking, 'Another stick. Great.' But wait... she's analyzing it! 'Looks like a 'gun'! A gun... made of moss and woodland vibes. It's a pacifist's firearm. It fires... good feelings? <giggle> And the 'thing to align you on the top'? What even is that? Is it for spirit levels? Is it a tiny, natural antenna? The questions are endless!"
TEST_VOICE = "tara"
# MODEL_PATH is NO LONGER USED HERE - orpheus_cpp downloads its own model
OUTPUT_PATH = "C:/_myDrive/repos/auto-vlog/Orpheus-TTS/output_test.wav" # Saving in the script's directory
GPU_LAYERS = -1  # Use -1 to offload all possible layers to GPU
VERBOSE_LOGGING = True # Set to True to see llama.cpp messages
TARGET_LANG = "en" # Explicitly set language to English

# --- Available Voices (for English) ---
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


def main():
    # --- Validate Hardcoded Parameters ---
    if TEST_VOICE not in AVAILABLE_VOICES:
         print(f"Warning: Hardcoded voice '{TEST_VOICE}' not in known list for English. Proceeding anyway.")
    if TARGET_LANG not in OrpheusCpp.lang_to_model:
         print(f"FATAL: Language '{TARGET_LANG}' is not supported by this version of OrpheusCpp.")
         sys.exit(1)

    print(f"Initializing OrpheusCpp for language '{TARGET_LANG}'...")
    print(f"GPU Layers: {GPU_LAYERS}")
    print(f"Verbose: {VERBOSE_LOGGING}")
    print(f"(Note: OrpheusCpp will download model: {OrpheusCpp.lang_to_model[TARGET_LANG]})")

    # Initialize OrpheusCpp - Pass ONLY valid arguments
    try:
        orpheus = OrpheusCpp(
            n_gpu_layers=GPU_LAYERS,
            verbose=VERBOSE_LOGGING,
            lang=TARGET_LANG
        )
        print("OrpheusCpp initialized successfully.")
    except Exception as init_e:
            print(f"FATAL: Error initializing OrpheusCpp: {init_e}")
            sys.exit(1)


    print(f"Generating speech for: '{TEST_TEXT}' with voice '{TEST_VOICE}'")
    start_time = time.time()

    # --- Streaming TTS ---
    buffer = []
    tts_options = {"voice_id": TEST_VOICE}
    sr = None # Initialize sample rate

    try:
        for i, (sr_chunk, chunk) in enumerate(orpheus.stream_tts_sync(TEST_TEXT, options=tts_options)):
            if sr is None and sr_chunk is not None: sr = sr_chunk
            # Ensure chunk is numpy array before buffer append
            if not isinstance(chunk, np.ndarray):
                 # The example returns bytes, let's convert based on that
                 chunk = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)

            buffer.append(chunk)
            if VERBOSE_LOGGING:
                print(f"Generated chunk {i}, Sample Rate: {sr_chunk}, Chunk Shape: {chunk.shape if hasattr(chunk, 'shape') else 'N/A'}")

    except Exception as stream_e:
        print(f"FATAL: Error during TTS streaming: {stream_e}")
        import traceback
        traceback.print_exc() # Print full traceback for streaming errors
        sys.exit(1)

    if not buffer:
            print("Error: No audio chunks generated.")
            return

    # Concatenate all chunks
    try:
        # Ensure all chunks are 2D (1, N) before concatenating axis=1
        processed_buffer = []
        for i, b in enumerate(buffer):
            if not isinstance(b, np.ndarray):
                 print(f"Error: Chunk {i} is not a numpy array (Type: {type(b)}). Skipping.")
                 continue
            if b.ndim == 1:
                 processed_buffer.append(b.reshape(1, -1)) # Reshape flat arrays
            elif b.ndim == 2 and b.shape[0] == 1:
                 processed_buffer.append(b) # Already correct shape
            else:
                 print(f"Warning: Skipping chunk {i} with unexpected shape {b.shape}.")
                 continue # Skip chunks with unexpected shapes

        if not processed_buffer:
             print("Error: No valid audio chunks to concatenate after processing shapes.")
             return

        # Now concatenate along axis=1 (the sample axis)
        full_audio_2d = np.concatenate(processed_buffer, axis=1)
        # Squeeze to make it flat (samples,) for writing
        full_audio = full_audio_2d.squeeze()
        if VERBOSE_LOGGING: print(f"Concatenated {len(processed_buffer)} chunks along axis=1, final shape: {full_audio.shape}")

    except Exception as concat_e:
         print(f"FATAL: Error concatenating audio chunks: {concat_e}")
         # Print shapes for debugging
         for i, b in enumerate(buffer): print(f"Original Chunk {i} Type: {type(b)}, Shape: {getattr(b, 'shape', 'N/A')}")
         sys.exit(1)

    # Get sample rate
    output_sample_rate = sr
    if not output_sample_rate:
        output_sample_rate = 24000 # Default if SR detection failed
        print(f"Warning: Could not detect sample rate from stream, defaulting to {output_sample_rate} Hz.")

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write the final audio (should be int16 now)
    try:
        if full_audio.dtype != np.int16:
            print(f"Warning: Final audio data type is {full_audio.dtype}, expected int16. Trying to cast.")
            full_audio = full_audio.astype(np.int16)

        print(f"Writing audio to: {OUTPUT_PATH} with sample rate {output_sample_rate} Hz")
        write(OUTPUT_PATH, output_sample_rate, full_audio)

        end_time = time.time()
        duration = len(full_audio) / output_sample_rate
        print(f"Audio generation complete in {end_time - start_time:.2f} seconds.")
        print(f"Generated {duration:.2f} seconds of audio.")
        print(f"Output saved to: {OUTPUT_PATH}")

    except Exception as write_e:
        print(f"FATAL: Error writing WAV file: {write_e}")
        sys.exit(1)


if __name__ == '__main__':
    main()