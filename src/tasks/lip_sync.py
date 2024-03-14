import os
import shutil
from pathlib import Path
from typing import Optional

import flytekit
from flytekit import ImageSpec, Resources, task
from flytekit.extras.accelerators import T4
from flytekit.types.file import FlyteFile

lip_sync_image = ImageSpec(
    name="lip_sync",
    builder="ucimage",
    apt_packages=["build-essential", "libssl-dev", "ffmpeg", "libsndfile1", "git"],
    packages=[
        "setuptools==69.1.1",
        "wheel",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "torchaudio==2.0.1",
        "numpy==1.23.5",
        "face-alignment==1.3.5",
        "imageio==2.27.0",
        "imageio-ffmpeg==0.4.8",
        "librosa==0.10.0.post2",
        "numba==0.59.0",
        "resampy==0.4.2",
        "pydub==0.25.1",
        "scipy==1.12.0",
        "kornia==0.6.11",
        "tqdm==4.65.0",
        "yacs==0.1.8",
        "pyyaml==6.0.1",
        "joblib==1.2.0",
        "scikit-image==0.20.0",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "dlib-bin==19.24.2",
        "gfpgan==1.3.8",
        "av==11.0.0",
        "safetensors==0.4.2",
        "huggingface-hub==0.21.4",
        "realesrgan==0.3.0",
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
    requests=(Resources(gpu="1", mem="30Gi")),
    container_image=lip_sync_image,
    accelerator=T4,
)
def lip_sync(
    audio_path: FlyteFile,
    pic_path: FlyteFile,
    ref_pose: FlyteFile,
    ref_eyeblink: FlyteFile,
    pose_style: int,
    batch_size: int,
    expression_scale: float,
    input_yaw_list: Optional[list[int]],
    input_pitch_list: Optional[list[int]],
    input_roll_list: Optional[list[int]],
    enhancer: str,
    background_enhancer: str,
    device: str,
    still: bool,
    preprocess: str,
    checkpoint_dir: str,
    size: int,
) -> FlyteFile:
    from src.lip_sync_src.facerender.animate import AnimateFromCoeff
    from src.lip_sync_src.generate_batch import get_data
    from src.lip_sync_src.generate_facerender_batch import get_facerender_data
    from src.lip_sync_src.test_audio2coeff import Audio2Coeff
    from src.lip_sync_src.utils.init_path import init_path
    from src.lip_sync_src.utils.preprocess import CropAndExtract

    audio_path = audio_path.download()
    pic_path = pic_path.download()

    if ref_eyeblink.remote_source == ref_pose.remote_source:
        ref_eyeblink = ref_eyeblink.download()
        ref_pose = ref_eyeblink
    else:
        ref_eyeblink = ref_eyeblink.download()
        ref_pose = ref_pose.download()

    working_dir = flytekit.current_context().working_directory
    save_dir = os.path.join(working_dir, "result")
    os.makedirs(save_dir, exist_ok=True)

    sadtalker_paths = init_path(
        checkpoint_dir,
        os.path.join("/root", "src", "lip_sync_src", "config"),
        size,
        preprocess,
    )

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)

    print("3DMM Extraction for source image")
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path,
        first_frame_dir,
        preprocess,
        source_image_flag=True,
        pic_size=size,
    )
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink != "":
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print("3DMM Extraction for the reference video providing eye blinking")
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            ref_eyeblink,
            ref_eyeblink_frame_dir,
            preprocess,
            source_image_flag=False,
        )
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose != "":
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing pose")
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False
            )
    else:
        ref_pose_coeff_path = None

    # audio2ceoff
    batch = get_data(
        first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still
    )
    coeff_path = audio_to_coeff.generate(
        batch, save_dir, pose_style, ref_pose_coeff_path
    )

    # coeff2video
    data = get_facerender_data(
        coeff_path,
        crop_pic_path,
        first_coeff_path,
        audio_path,
        batch_size,
        input_yaw_list,
        input_pitch_list,
        input_roll_list,
        expression_scale=expression_scale,
        still_mode=still,
        preprocess=preprocess,
        size=size,
    )

    result = animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=enhancer,
        background_enhancer=background_enhancer,
        preprocess=preprocess,
        img_size=size,
    )

    file_path = Path(save_dir, "output.mp4").as_posix()

    shutil.move(result, file_path)
    print("The generated video is named: ", file_path)

    return FlyteFile(file_path)
