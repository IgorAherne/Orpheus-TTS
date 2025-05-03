import asyncio
import platform
import sys
import threading
import os # Added for path checks
from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Literal,
    cast,
)

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

DEFAULT_SAMPLE_RATE = 24000

# Constants
# Voice definitions per language (Expand as needed based on models)
LANG_VOICES = {
    "en": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
    "es": ["javi", "sergio", "maria"], # Example Spanish voices
    "it": ["pietro", "giulia", "carlo"], # Example Italian voices
    "fr": ["pierre", "amelie", "marie"],
    "de": ["jana", "thomas", "max"],
    "ko": ["유나", "준서"], # Ensure your terminal/editor supports UTF-8
    "hi": ["ऋतिका"],
    "zh": ["长乐", "白芷"],
    # Add other languages and their voices here
}

# Define available languages and their default models
# Using the structure from the original class
LANG_TO_MODEL_REPO = {
    "en": "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF", # Example, adjust if needed
    "es": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
    "fr": "freddyaboulton/3b-fr-ft-research_release-Q4_K_M-GGUF",
    "de": "freddyaboulton/3b-de-ft-research_release-Q4_K_M-GGUF",
    "it": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF", # Shared with es
    "hi": "freddyaboulton/3b-hi-ft-research_release-Q4_K_M-GGUF",
    "zh": "freddyaboulton/3b-zh-ft-research_release-Q4_K_M-GGUF",
    "ko": "freddyaboulton/3b-ko-ft-research_release-Q4_K_M-GGUF",
}
# Define a high-quality Q8 model path for English as a common local default
DEFAULT_LOCAL_EN_MODEL = r"C:/_myDrive/repos/auto-vlog/models/GGUF/Orpheus-3b-FT-Q8_0.gguf"


class TTSOptions(TypedDict):
    max_tokens: NotRequired[int] #Maximum number of tokens to generate. Default: 65536

    temperature: NotRequired[float] #Temperature for sampling. Default: 0.8
    
    top_p: NotRequired[float] # Top-p sampling. Default: 0.95 
    
    top_k: NotRequired[int] # Top-k sampling. Default: 40 
    
    min_p: NotRequired[float] # Minimum probability for top-p sampling. Default: 0.05 
    
    repeat_penalty: NotRequired[float] # Penalty for repeating tokens. Default: 1.1 
    
    pre_buffer_size: NotRequired[float] # Seconds of audio to generate before yielding the first chunk. Default: 1.5 
    
    voice_id: NotRequired[str] # The voice ID to use. Varies by language. 


CUSTOM_TOKEN_PREFIX = "<custom_token_"


class OrpheusCpp:
    # Make the mapping available as a class attribute
    lang_to_model = LANG_TO_MODEL_REPO

    def __init__(
        self,
        model_path: str | None = None, # Allow specifying a path directly
        lang: str = "en", # Default language if path is not specified
        n_gpu_layers: int = -1, # Default to max GPU layers
        n_threads: int = 0,
        verbose: bool = True,
    ):
        # Import llama_cpp dynamically
        try:
            from llama_cpp import Llama
        except ImportError:
            # (Keep the existing platform-specific installation instructions)
            if sys.platform == "darwin":
                 is_arm64 = platform.machine() == "arm64"
                 version = platform.mac_ver()[0].split(".")
                 is_macos_11_plus = len(version) >= 2 and int(version[0]) >= 11
                 is_macos_10_less = len(version) >= 2 and int(version[0]) < 11

                 if is_arm64 and is_macos_11_plus:
                    extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal"
                 elif is_macos_10_less:
                     raise ImportError(...) # Original message
                 else:
                     extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            else:
                 extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            raise ImportError(
                f"llama_cpp is not installed. Please install it using `pip install llama-cpp-python {extra_index_url}`."
            )

        # Determine Model Path
        final_model_path = None
        if model_path:
            print(f"Attempting to use provided model path: {model_path}")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Provided model path does not exist or is not a file: {model_path}")
            final_model_path = model_path
        elif lang == "en" and os.path.isfile(DEFAULT_LOCAL_EN_MODEL):
             print(f"Using default local English model: {DEFAULT_LOCAL_EN_MODEL}")
             final_model_path = DEFAULT_LOCAL_EN_MODEL
        else:
            if lang not in self.lang_to_model:
                raise ValueError(f"Language '{lang}' not supported or no default model defined. Available: {list(self.lang_to_model.keys())}")
            repo_id = self.lang_to_model[lang]
            # Infer filename convention (repo name, lowercase, .gguf)
            filename = repo_id.split("/")[-1].lower().replace("-gguf", ".gguf")
            print(f"No specific model path provided. Attempting to download/cache model for lang '{lang}'...")
            print(f"Repository: {repo_id}, Filename: {filename}")
            try:
                # Use local_files_only=True first to prefer cache? Or check manually?
                # For simplicity, let hf_hub_download handle caching.
                final_model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    # You might want to add:
                    # cache_dir="path/to/your/model_cache"
                    # local_files_only=False, # Set to True to prevent downloads if not cached
                )
                print(f"Using model from cache/download: {final_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download/find model for lang '{lang}' from '{repo_id}': {e}")

        self.loaded_model_path = final_model_path # Store the path actually used

        # Initialize Llama
        print("Initializing Llama model...")
        self._llm = Llama(
            model_path=self.loaded_model_path,
            n_ctx=32768, # Increased context based on console log, adjust if needed
            verbose=verbose,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            # batch_size=1, # llama-cpp-python handles batching internally for __call__ stream
            # logits_all=True, # Might be needed by some sampling methods, but default is False
        )
        print(f"Llama model loaded from: {self.loaded_model_path}")

        # Initialize SNAC
        print("Initializing SNAC vocoder...")
        repo_id_snac = "onnx-community/snac_24khz-ONNX"
        snac_model_file = "decoder_model.onnx"
        try:
            snac_model_path = hf_hub_download(
                repo_id_snac, subfolder="onnx", filename=snac_model_file
            )
            print(f"SNAC model path: {snac_model_path}")

            # Determine available providers
            available_providers = onnxruntime.get_available_providers()
            providers_to_try = []
            if "CUDAExecutionProvider" in available_providers:
                providers_to_try.append("CUDAExecutionProvider")
            providers_to_try.append("CPUExecutionProvider") # Always include CPU as fallback

            print(f"Attempting to load SNAC with providers: {providers_to_try}")
            self._snac_session = onnxruntime.InferenceSession(
                snac_model_path,
                providers=providers_to_try,
            )
            print(f"SNAC session loaded with provider: {self._snac_session.get_providers()}")
        except Exception as e:
             raise RuntimeError(f"Failed to load SNAC model: {e}")


    def _token_to_id(self, token_text: str, index: int) -> int | None:
        # (Keep the existing _token_to_id implementation)
        token_string = token_text.strip()
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
        if last_token_start == -1: return None
        last_token = token_string[last_token_start:]
        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError: return None
        else: return None


    def _decode(
        self, token_gen: Generator[str, None, None]
    ) -> Generator[np.ndarray, None, None]:
        # (Keep the existing _decode implementation)
        buffer = []
        count = 0
        for token_text in token_gen:
            token = self._token_to_id(token_text, count)
            if token is None or token <= 0: continue
            
            buffer.append(token)
            count += 1
            if count % 7 != 0 or count <= 27: continue

            buffer_to_proc = buffer[-28:]
            audio_samples = self._convert_to_audio(buffer_to_proc)
            if audio_samples is None: continue

            yield audio_samples


    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        # (Keep the existing _convert_to_audio implementation)
        if len(multiframe) < 28: return None
        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]
        codes_0 = np.array([], dtype=np.int32)
        codes_1 = np.array([], dtype=np.int32)
        codes_2 = np.array([], dtype=np.int32)
        for j in range(num_frames):
            i = 7 * j
            codes_0 = np.append(codes_0, frame[i])
            codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])
            codes_2 = np.append(codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])
        codes_0 = np.expand_dims(codes_0, axis=0)
        codes_1 = np.expand_dims(codes_1, axis=0)
        codes_2 = np.expand_dims(codes_2, axis=0)
        if ( np.any(codes_0 < 0) or np.any(codes_0 > 4096) or
             np.any(codes_1 < 0) or np.any(codes_1 > 4096) or
             np.any(codes_2 < 0) or np.any(codes_2 > 4096) ):
            # print("Warning: Skipping frame due to out-of-range token values.") # Optional: Add logging
            return None
        snac_input_names = [x.name for x in self._snac_session.get_inputs()]
        input_dict = dict(zip(snac_input_names, [codes_0, codes_1, codes_2]))
        try:
            audio_hat = self._snac_session.run(None, input_dict)[0]
            audio_np = audio_hat[:, :, 2048:4096] # Extract relevant part
            # Ensure output is float before scaling and casting
            if audio_np.dtype != np.float32:
                 audio_np = audio_np.astype(np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            # Return the ndarray directly, not bytes
            return audio_int16 # Shape should be (1, N)
        except Exception as e:
            print(f"Error during SNAC inference: {e}")
            return None


    def _token_gen(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[str, None, None]:
        from llama_cpp import CreateCompletionStreamResponse, CompletionChunk

        options = options or TTSOptions()
        voice_id = options.get("voice_id", "tara") # Default if not provided
        # Standard prompt format
        prompt = f"<|audio|>{voice_id}: {text}<|eot_id|><custom_token_4>"

        # Prepare generation arguments
        generation_params = {
            "max_tokens": options.get("max_tokens", 65536), # Use a larger default potentially
            "stream": True,
            "temperature": options.get("temperature", 0.8),
            "top_p": options.get("top_p", 0.95),
            "top_k": options.get("top_k", 40),
            "min_p": options.get("min_p", 0.05),
            "repeat_penalty": options.get("repeat_penalty", 1.1), # Added repeat_penalty
            # stop=["<|eot_id|>"], # Optional: Can add stop tokens if needed
        }

        token_stream = self._llm(prompt, **generation_params)

        # Correctly iterate over the stream response, ensuring 'text' exists in the choice dictionary
        for chunk in cast(Iterator[CompletionChunk], token_stream):
             if chunk["choices"] and "text" in chunk["choices"][0]:
                  yield chunk["choices"][0]["text"]
             # else: handle cases where text might be missing, if necessary


    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        options = options or TTSOptions()
        token_gen = self._token_gen(text, options)
        pre_buffer = np.array([], dtype=np.int16).reshape(1, 0) # Ensure 2D
        # Use frames instead of time for pre-buffer size for consistency
        pre_buffer_target_samples = int(DEFAULT_SAMPLE_RATE * options.get("pre_buffer_size", 1.5))
        started_yielding = False

        for audio_array_chunk in self._decode(token_gen):
            # _decode now returns np.int16 array directly, shape (1, N)
            if audio_array_chunk is None or audio_array_chunk.size == 0:
                continue

            if not started_yielding:
                pre_buffer = np.concatenate([pre_buffer, audio_array_chunk], axis=1)
                if pre_buffer.shape[1] >= pre_buffer_target_samples:
                    yield (DEFAULT_SAMPLE_RATE, pre_buffer)
                    started_yielding = True
                    pre_buffer = np.array([], dtype=np.int16).reshape(1, 0) # Clear buffer
            else:
                yield (DEFAULT_SAMPLE_RATE, audio_array_chunk)

        # Yield any remaining data in the pre_buffer if we never started yielding
        if not started_yielding and pre_buffer.shape[1] > 0:
            yield (DEFAULT_SAMPLE_RATE, pre_buffer)

    # Non-streaming methods (can be simplified or removed if only streaming is needed)
    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        """Generates audio non-streamingly."""
        buffer = []
        final_sr = DEFAULT_SAMPLE_RATE
        for sr, array_chunk in self.stream_tts_sync(text, options):
            final_sr = sr # Capture the sample rate
            buffer.append(array_chunk)

        if not buffer:
            return (final_sr, np.array([], dtype=np.int16)) # Return empty array if no chunks

        # Concatenate requires consistent shapes, stream_tts_sync now returns (1, N)
        full_audio = np.concatenate(buffer, axis=1)
        return (final_sr, full_audio.squeeze()) # Return 1D array

    # async stream_tts remains complex due to threading. If not strictly needed,
    # focusing on stream_tts_sync might be simpler for this specific use case.
    # Keeping it for completeness, but ensure it aligns with stream_tts_sync changes.
    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        """Asynchronous wrapper for stream_tts_sync using a queue."""
        queue = asyncio.Queue()
        finished = object() # Use a sentinel object to signal completion

        def _run_sync_in_thread():
            try:
                for chunk_tuple in self.stream_tts_sync(text, options):
                    queue.put_nowait(chunk_tuple)
            except Exception as e:
                # Propagate exception to the async loop
                queue.put_nowait(e)
            finally:
                # Signal completion
                queue.put_nowait(finished)

        thread = threading.Thread(target=_run_sync_in_thread, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is finished:
                break
            elif isinstance(item, Exception):
                thread.join() # Ensure thread finishes before raising
                raise item # Re-raise the exception in the async context
            else:
                yield item # Yield the (sr, chunk) tuple
            queue.task_done() # Mark task as done for queue management

        thread.join() # Wait for thread completion
