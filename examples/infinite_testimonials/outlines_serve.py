import argparse
import json
from typing import AsyncGenerator
import os 
from modal import Image, Secret, App, gpu, asgi_app
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

MODEL_DIR = "/model"
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

# Create a Modal image with the required dependencies
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "outlines",
        "vllm>=0.3.0",
        "pydantic>=2.0",
        "hf_transfer",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[Secret.from_dotenv()],
        timeout=60 * 20,
    )
)
# Create a Modal app
app = App("outlines-phi", image=image)
GPU_CONFIG = gpu.H100()

# Create a stub for the FastAPI app
web_app = FastAPI()

# Global variables
engine = None
tokenizer = None

@web_app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@web_app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request."""
    
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
    from outlines.processors import JSONLogitsProcessor, RegexLogitsProcessor
    global engine, tokenizer
    assert engine is not None

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    json_schema = request_dict.pop("schema", None)
    regex_string = request_dict.pop("regex", None)
    if json_schema is not None:
        logits_processors = [JSONLogitsProcessor(json_schema, tokenizer)]
    elif regex_string is not None:
        logits_processors = [RegexLogitsProcessor(regex_string, tokenizer)]
    else:
        logits_processors = []

    sampling_params = SamplingParams(
        **request_dict, logits_processors=logits_processors
    )
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

@app.function(
        allow_concurrent_inputs=1000,
        gpu=GPU_CONFIG,
        timeout=300,
        secrets=[
            Secret.from_dotenv(),
        ],)
@asgi_app()
def serve_vllm():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from transformers import SPIECE_UNDERLINE, PreTrainedTokenizerBase
    from typing import Union
    global engine, tokenizer
    
    def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
        """Adapt a tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of `transformers`. In
        addition we need to handle the missing spaces to Llama's tokenizer to be able to
        compile FSMs for this model.

        Parameters
        ----------
        tokenizer
            The tokenizer of the model.

        Returns
        -------
        PreTrainedTokenizerBase
            The adapted tokenizer.
        """
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: Union[str, bytes]) -> str:
            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if (
                type(token) is str
                and token.startswith(SPIECE_UNDERLINE)
                or token == "<0x20>"
            ):
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string

        return tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    #parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args, _ = parser.parse_known_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = adapt_tokenizer(tokenizer=engine.engine.tokenizer.tokenizer)

    return web_app