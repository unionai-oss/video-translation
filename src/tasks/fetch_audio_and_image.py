import os
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
    packages=["moviepy==1.0.3", "katna==0.9.2"],
)

if preprocessing_image.is_container():
    from Katna.video import Video
    from Katna.writer import KeyFrameDiskWriter
    from moviepy.editor import VideoFileClip


def extract_random_frame(video_path, image_dir):
    # initialize video module
    vd = Video()

    # number of images to be returned
    no_of_frames_to_return = 1

    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=image_dir)

    print(video_path)

    print(f"Input video file path = {video_path}")

    # extract the best keyframe and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_return,
        file_path=video_path,
        writer=diskwriter,
    )


audio_and_image_values = NamedTuple(
    "audio_and_image_values", [("audio", FlyteFile), ("image", FlyteFile)]
)


@task(
    cache=True,
    cache_version="2",
    container_image=preprocessing_image,
    requests=Resources(mem="5Gi", cpu="1"),
    accelerator=T4,
)
def fetch_audio_and_image(
    video_file: FlyteFile, output_ext: str
) -> audio_and_image_values:
    downloaded_video = video_file.download()

    # AUDIO
    video_filename, _ = os.path.splitext(downloaded_video)
    clip = VideoFileClip(downloaded_video)

    audio_file_path = Path(
        flytekit.current_context().working_directory, f"{video_filename}.{output_ext}"
    ).as_posix()
    clip.audio.write_audiofile(audio_file_path)

    # IMAGE
    if os.path.splitext(downloaded_video)[1] == "":
        new_file_name = downloaded_video + ".mp4"
        os.rename(downloaded_video, new_file_name)
        downloaded_video = new_file_name

    image_dir = flytekit.current_context().working_directory
    extract_random_frame(downloaded_video, image_dir)

    return audio_and_image_values(
        audio=FlyteFile(audio_file_path),
        image=FlyteFile(Path(image_dir, os.listdir(image_dir)[0]).as_posix()),
    )
