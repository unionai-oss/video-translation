import os

import numpy as np
import requests
from flytekit import ImageSpec, Resources, task
from flytekit.extras.accelerators import T4
from flytekit.types.file import FlyteFile

speech2text_image = ImageSpec(
    name="speech2text",
    builder="ucimage",
    apt_packages=["ffmpeg"],
    packages=[
        "transformers==4.36.2",
        "torch==2.2.1",
        "flytekit==1.10.7",
        "unionai==0.1.5",
    ],
    cuda="12.1.0",
    cudnn="8",
    python_version="3.11",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


@task(
    cache=True,
    cache_version="2",
    container_image=speech2text_image,
    requests=Resources(gpu="1", mem="10Gi", cpu="1"),
    accelerator=T4,
)
def speech2text(
    checkpoint: str,
    audio: FlyteFile,
    chunk_length: float,
    return_timestamps: bool,
    translate_from: str,
) -> str:
    import torch
    from transformers import pipeline
    from transformers.pipelines.audio_utils import ffmpeg_read

    pipe = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        chunk_length_s=chunk_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    local_audio_path = audio.download()
    if local_audio_path.startswith("http://") or local_audio_path.startswith(
        "https://"
    ):
        inputs = requests.get(inputs).content
    else:
        with open(local_audio_path, "rb") as f:
            inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")

    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
        )

    prediction = pipe(
        inputs,
        return_timestamps=return_timestamps,
        generate_kwargs={"task": "transcribe", "language": translate_from},
    )
    output = prediction["text"].strip()

    return output
