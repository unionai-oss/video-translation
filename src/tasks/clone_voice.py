from pathlib import Path

import flytekit
from flytekit import ImageSpec, Resources, task
from flytekit.extras.accelerators import T4
from flytekit.types.file import FlyteFile

language_codes = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}

clone_voice_image = ImageSpec(
    name="clone_voice",
    builder="ucimage",
    packages=[
        "TTS==0.22.0",
        "flytekit==1.10.7",
        "unionai==0.1.5",
    ],
    cuda="12.1.0",
    cudnn="8",
    python_version="3.11",
)


@task(
    cache=True,
    cache_version="2",
    container_image=clone_voice_image,
    requests=Resources(gpu="1", mem="15Gi"),
    accelerator=T4,
    environment={"COQUI_TOS_AGREED": "1"},
)
def clone_voice(text: str, target_lang: str, speaker_wav: FlyteFile) -> FlyteFile:
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    file_path = Path(
        flytekit.current_context().working_directory, "output.wav"
    ).as_posix()

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav.download(),
        language=language_codes[target_lang],
        file_path=file_path,
        split_sentences=True,
    )
    return FlyteFile(file_path)
