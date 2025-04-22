import os
import sys
import time
import argparse
import numpy as np
from scipy.io.wavfile import write
import onnxruntime # Import early for provider check
import traceback

# Local Implementation Import
try:
    from orpheus_impl import OrpheusCpp, TTSOptions, LANG_TO_MODEL_REPO
except ImportError as e:
    print(f"FATAL: Failed to import OrpheusCpp from orpheus_impl.py: {e}")
    sys.exit(1)


DEFAULT_SAMPLE_RATE = 24000

# Helper Functions (keep check_onnxruntime_gpu and check_cuda_path)
def check_onnxruntime_gpu():
    """Checks for ONNXRuntime CUDA provider and warns if unavailable."""
    available_providers = onnxruntime.get_available_providers()
    print("--- ONNXRuntime Provider Check")
    print(f"Available Providers: {available_providers}")
    if "CUDAExecutionProvider" in available_providers:
        print("INFO: ONNXRuntime CUDA Provider is available.")
        return True
    print("")
    print(" WARNING: ONNXRuntime CUDA Provider unavailable.")
    print(" SNAC model will run on CPU (slower).           ")
    print(" Install `onnxruntime-gpu` for GPU acceleration.")
    print("")
    return False


def check_cuda_path():
    """Checks if CUDA_PATH is set in the environment."""
    cuda_path_env = os.environ.get('CUDA_PATH')
    print("--- Environment Check")
    print(f"CUDA_PATH: {cuda_path_env if cuda_path_env else 'Not Set'}")
    # Removed warning as presence isn't strictly required if drivers/libs are found otherwise


# Main Execution Logic
def run_tts(args):
    """Initializes Orpheus, generates audio, and saves the output."""
    check_cuda_path()
    check_onnxruntime_gpu()

    # Validate Language and Voice
    selected_lang = args.lang
    if args.model_path:
        # If model path is given, language selection is bypassed for model loading,
        # but we still need it for voice validation. Default to 'en' if unsure.
        # A more robust solution might try to infer lang from model name/path.
        if selected_lang not in LANG_VOICES:
            print(f"Warning: Language '{selected_lang}' not explicitly defined for voice validation when using --model-path. Defaulting to 'en' voices.")
            selected_lang_for_voices = 'en'
        else:
            selected_lang_for_voices = selected_lang
        print(f"Note: Using provided model path. Language '{selected_lang}' used for voice validation.")
    else:
        if selected_lang not in LANG_VOICES:
             print(f"Error: Language '{selected_lang}' not supported or has no defined voices. Available: {list(LANG_VOICES.keys())}")
             sys.exit(1)
        selected_lang_for_voices = selected_lang

    available_voices = LANG_VOICES.get(selected_lang_for_voices, [])
    if not available_voices:
         print(f"Error: No voices defined for language '{selected_lang_for_voices}'.")
         sys.exit(1)
    if args.voice not in available_voices:
         print(f"Error: Voice '{args.voice}' is not valid for language '{selected_lang_for_voices}'. Available: {available_voices}")
         # Suggest default if voice is invalid but lang is ok
         print(f"Try using the default voice: '{available_voices[0]}'")
         sys.exit(1)

    print("\n--- Initializing OrpheusCpp")
    if args.model_path:
        print(f"Using specified model path: {args.model_path}")
    else:
        print(f"Using language-based model selection for: '{args.lang}'")
    print(f"GPU Layers: {args.gpu_layers} | Threads: {args.threads if args.threads > 0 else 'Auto'} | Verbose: {args.verbose}")

    try:
        # Pass model_path=None if not specified, so impl uses lang
        orpheus = OrpheusCpp(
            model_path=args.model_path, # Pass the specific path or None
            lang=args.lang,             # Pass the language regardless (used if model_path is None)
            n_gpu_layers=args.gpu_layers,
            n_threads=args.threads,
            verbose=args.verbose,
        )
        print(f"OrpheusCpp initialized successfully. Loaded model: {orpheus.loaded_model_path}")
    except Exception as init_e:
        print(f"FATAL: Error initializing OrpheusCpp: {init_e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n--- Starting TTS Generation")
    print(f"Lang: {args.lang} | Voice: {args.voice} | Output: {args.output}")
    if args.verbose: print(f"Text: '{args.text}'")
    start_time = time.time()

    tts_options: TTSOptions = {
        "voice_id": args.voice,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repeat_penalty": args.repeat_penalty, # Added repeat penalty
        "pre_buffer_size": args.pre_buffer_size,
    }
    if args.verbose: print(f"TTS Options: {tts_options}")

    # Streaming and Buffering (Identical to previous tidy version)
    audio_buffer = []
    output_sample_rate = None
    chunk_count = 0
    try:
        for i, (sr_chunk, chunk_data) in enumerate(orpheus.stream_tts_sync(args.text, options=tts_options)):
            if output_sample_rate is None and sr_chunk is not None:
                output_sample_rate = sr_chunk
                if args.verbose: print(f"Detected Sample Rate: {output_sample_rate} Hz")

            if not isinstance(chunk_data, np.ndarray) or chunk_data.size == 0:
                if args.verbose: print(f"Skipping invalid chunk {i}")
                continue

            try:
                if chunk_data.dtype != np.int16: chunk_data = chunk_data.astype(np.int16)
                if chunk_data.ndim == 1: chunk_data_processed = chunk_data.reshape(1, -1)
                elif chunk_data.ndim == 2 and chunk_data.shape[0] == 1: chunk_data_processed = chunk_data
                else: raise ValueError(f"Unexpected chunk shape {chunk_data.shape}")

                audio_buffer.append(chunk_data_processed)
                chunk_count += 1
                if args.verbose and i % 100 == 0: print(f"Processed chunk {i}...") # Log less often

            except Exception as chunk_proc_e:
                 print(f"Warning: Error processing chunk {i}: {chunk_proc_e}. Skipping.")

        if chunk_count == 0:
            print("Error: No valid audio chunks were generated.")
            sys.exit(1)
        if args.verbose: print(f"Finished streaming. Processed {chunk_count} valid chunks.")

    except Exception as stream_e:
        print(f"FATAL: Error during TTS streaming: {stream_e}")
        traceback.print_exc()
        sys.exit(1)

    # Concatenation and Saving (Identical to previous tidy version)
    if not audio_buffer:
        print("Error: Audio buffer is empty.")
        sys.exit(1)

    try:
        full_audio_2d = np.concatenate(audio_buffer, axis=1)
        full_audio_1d = full_audio_2d.squeeze()

        if full_audio_1d.ndim != 1: raise ValueError(f"Concatenated audio has unexpected dimensions: {full_audio_1d.ndim}")
        if full_audio_1d.dtype != np.int16: full_audio_1d = full_audio_1d.astype(np.int16)

        final_sr = output_sample_rate or DEFAULT_SAMPLE_RATE
        if not output_sample_rate: print(f"Warning: Sample rate not detected, using default {DEFAULT_SAMPLE_RATE} Hz.")

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

        print(f"Writing audio to: {args.output} (Sample Rate: {final_sr} Hz)")
        write(args.output, final_sr, full_audio_1d)

        end_time = time.time()
        generation_time = end_time - start_time
        duration = len(full_audio_1d) / final_sr
        rtf = generation_time / duration if duration > 0 else float('inf')

        print("\n--- Generation Complete")
        print(f"Output: {args.output}")
        print(f"Audio Duration: {duration:.2f}s")
        print(f"Generation Time: {generation_time:.2f}s")
        print(f"Realtime Factor (RTF): {rtf:.2f}")

    except Exception as post_proc_e:
        print(f"FATAL: Error during audio concatenation or saving: {post_proc_e}")
        traceback.print_exc()
        sys.exit(1)


# Argument Parser Setup
def setup_arg_parser():
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate speech using Orpheus TTS via orpheus_impl.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    # Core Arguments
    parser.add_argument("text", help="Text to synthesize.")
    parser.add_argument("-o", "--output", default="output_orpheus.wav", help="Output WAV file path.")

    # Model Selection Arguments
    parser.add_argument("--lang", default="en", choices=LANG_VOICES.keys(), help="Language for voice and automatic model selection (if --model-path is not set).")
    parser.add_argument("--model-path", default=None, help="Specify a direct path to a GGUF model file. Overrides --lang for model selection.")
    parser.add_argument("--voice", default=None, help="Voice ID to use. If not set, defaults to the first available voice for the selected language.")

    # Hardware Configuration
    parser.add_argument("--gpu-layers", type=int, default=-1, help="LLM layers to offload to GPU (-1 for all possible).")
    parser.add_argument("--threads", type=int, default=0, help="LLM CPU threads (0 for auto).")

    # Generation Parameters
    parser.add_argument("--max-tokens", type=int, default=65536, help="Max tokens for LLM generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling.")
    parser.add_argument("--min-p", type=float, default=0.05, help="Minimum probability sampling.")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Penalty for token repetition.") # Added
    parser.add_argument("--pre-buffer-size", type=float, default=1.5, help="Audio buffer duration (seconds) before yielding.")
    # Other
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")

    return parser


# Script Entry Point
if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    cli_args = arg_parser.parse_args()

    # Set default voice if not provided
    if cli_args.voice is None:
        default_voice_list = LANG_VOICES.get(cli_args.lang)
        if default_voice_list:
            cli_args.voice = default_voice_list[0]
            print(f"Voice not specified, defaulting to '{cli_args.voice}' for language '{cli_args.lang}'.")
        else:
            # This case should be caught later, but added for safety
            print(f"Error: Cannot determine default voice for language '{cli_args.lang}'. Please specify --voice.")
            sys.exit(1)

    # Validate model path existence if provided
    if cli_args.model_path and not os.path.isfile(cli_args.model_path):
         print(f"Error: Specified --model-path does not exist or is not a file: {cli_args.model_path}")
         sys.exit(1)

    run_tts(cli_args)

# END OF FILE orpheus_tts_tidy.py (Modified)