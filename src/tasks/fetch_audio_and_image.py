import os
import random
from pathlib import Path
from typing import NamedTuple

import flytekit
from flytekit import ImageSpec, Resources, task
from flytekit.extras.accelerators import T4
from flytekit.types.file import FlyteFile

preprocessing_image = ImageSpec(
    name="fetch_audio_and_image",
    registry="samhitaalla",
    apt_packages=["ffmpeg"],
    packages=["moviepy==1.0.3", "opencv-python==4.9.0.80"],
)

if preprocessing_image.is_container():
    import cv2
    from moviepy.editor import VideoFileClip


def extract_random_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_number = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    _, frame = cap.read()
    cap.release()
    cv2.imwrite(output_image_path, frame)


audio_and_image_values = NamedTuple(
    "audio_and_image_values", [("audio", FlyteFile), ("image", FlyteFile)]
)


@task(
    cache=True,
    cache_version="1",
    container_image=preprocessing_image,
    requests=Resources(mem="5Gi", cpu="1"),
    accelerator=T4,
)
def fetch_audio_and_image(
    video_file: FlyteFile, output_ext: str
) -> audio_and_image_values:
    # AUDIO
    downloaded_video = video_file.download()
    video_filename, _ = os.path.splitext(downloaded_video)
    clip = VideoFileClip(downloaded_video)

    audio_file_path = Path(
        flytekit.current_context().working_directory, f"{video_filename}.{output_ext}"
    ).as_posix()
    clip.audio.write_audiofile(audio_file_path)

    # IMAGE
    image_file_path = Path(
        flytekit.current_context().working_directory, "image.jpg"
    ).as_posix()
    extract_random_frame(downloaded_video, image_file_path)

    return audio_and_image_values(
        audio=FlyteFile(audio_file_path), image=FlyteFile(image_file_path)
    )
