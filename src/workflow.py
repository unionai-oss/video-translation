from typing import Optional

from flytekit import workflow
from flytekit.types.file import FlyteFile

from .tasks.clone_voice import clone_voice
from .tasks.fetch_audio_and_image import fetch_audio_and_image
from .tasks.lip_sync import lip_sync
from .tasks.speech2text import speech2text
from .tasks.translate_text import translate_text


@workflow
def video_translation_wf(
    video_file: FlyteFile = "https://github.com/Zz-ww/SadTalker-Video-Lip-Sync/raw/master/sync_show/original.mp4",
    translate_from: str = "English",
    translate_to: str = "German",
    checkpoint: str = "openai/whisper-large-v2",
    output_ext: str = "mp3",
    chunk_length: float = 30.0,
    return_timestamps: bool = False,
    ref_pose: FlyteFile = "https://github.com/Zz-ww/SadTalker-Video-Lip-Sync/raw/master/sync_show/original.mp4",
    ref_eyeblink: FlyteFile = "https://github.com/Zz-ww/SadTalker-Video-Lip-Sync/raw/master/sync_show/original.mp4",
    pose_style: int = 0,
    batch_size: int = 2,
    expression_scale: float = 1.0,
    input_yaw_list: Optional[list[int]] = None,
    input_pitch_list: Optional[list[int]] = None,
    input_roll_list: Optional[list[int]] = None,
    enhancer: str = "gfpgan",
    background_enhancer: str = "",
    device: str = "cuda",
    still: bool = True,
    preprocess: str = "extfull",
    size: int = 512,
    checkpoint_dir: str = "vinthony/SadTalker-V002rc",  # HF model
) -> FlyteFile:
    """
    Video translation Flyte workflow.

    :param video_file: The video file to translate.
    :param translate_from: The language to translate from.
    :param translate_to: The language to translate to, options are ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Polish', 'Turkish', 'Russian', 'Dutch', 'Czech', 'Arabic', 'Chinese', 'Japanese', 'Hungarian', 'Korean', 'Hindi']
    :param checkpoint: Speech-to-text model checkpoint.
    :param output_ext: Output extension for audio files.
    :param chunk_length: Length of audio chunks.
    :param return_timestamps: If set to True, provides start and end timestamps for each recognized word or segment in the output, along with the transcribed text.
    :param ref_pose: Path to reference video providing pose.
    :param ref_eyeblink: Path to reference video providing eye blinking.
    :param pose_style: Input pose style from [0, 46).
    :param batch_size: Batch size of facerender.
    :param expression_scale: A larger value will make the expression motion stronger.
    :param input_yaw_list: The input yaw degree of the user.
    :param input_pitch_list: The input pitch degree of the user.
    :param input_roll_list: The input roll degree of the user.
    :param enhancer: Face enhancer options include [gfpgan, RestoreFormer].
    :param background_enhancer: Background enhancer options include [realesrgan].
    :param device: The device to use, CPU or GPU.
    :param still: Can crop back to the original videos for the full body animation.
    :param preprocess: How to preprocess the images, options are ['crop', 'extcrop', 'resize', 'full', 'extfull'].
    :param size: The image size of the facerender, options are [256, 512].
    :param checkpoint_dir: Path to model checkpoint, currently hosted in a Hugging Face repository.
    """

    values = fetch_audio_and_image(video_file=video_file, output_ext=output_ext)
    text = speech2text(
        checkpoint=checkpoint,
        audio=values.audio,
        chunk_length=chunk_length,
        return_timestamps=return_timestamps,
        translate_from=translate_from,
    )
    translated_text = translate_text(
        translate_from=translate_from, translate_to=translate_to, input=text
    )
    cloned_voice = clone_voice(
        text=translated_text, target_lang=translate_to, speaker_wav=values.audio
    )
    return lip_sync(
        audio_path=cloned_voice,
        pic_path=values.image,
        ref_pose=ref_pose,
        ref_eyeblink=ref_eyeblink,
        pose_style=pose_style,
        batch_size=batch_size,
        expression_scale=expression_scale,
        input_yaw_list=input_yaw_list,
        input_pitch_list=input_pitch_list,
        input_roll_list=input_roll_list,
        enhancer=enhancer,
        background_enhancer=background_enhancer,
        device=device,
        still=still,
        preprocess=preprocess,
        size=size,
        checkpoint_dir=checkpoint_dir,
    )
