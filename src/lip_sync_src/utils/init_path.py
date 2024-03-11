import glob
import os

from huggingface_hub import hf_hub_download


def init_path(checkpoint_dir, config_dir, size, preprocess="crop"):
    #### load all the checkpoint of `pth`
    print("using safetensor as default")
    sadtalker_paths = {
        "checkpoint": hf_hub_download(
            repo_id=checkpoint_dir,
            filename="SadTalker_V0.0.2_" + str(size) + ".safetensors",
        ),
    }
    use_safetensor = True

    sadtalker_paths["dir_of_BFM_fitting"] = os.path.join(config_dir)  # , 'BFM_Fitting'
    sadtalker_paths["audio2pose_yaml_path"] = os.path.join(
        config_dir, "auido2pose.yaml"
    )
    sadtalker_paths["audio2exp_yaml_path"] = os.path.join(config_dir, "auido2exp.yaml")
    sadtalker_paths[
        "use_safetensor"
    ] = use_safetensor  # os.path.join(config_dir, 'auido2exp.yaml')

    if "full" in preprocess:
        sadtalker_paths["mappingnet_checkpoint"] = hf_hub_download(
            repo_id=checkpoint_dir,
            filename="mapping_00109-model.pth.tar",
        )
        sadtalker_paths["facerender_yaml"] = os.path.join(
            config_dir, "facerender_still.yaml"
        )
    else:
        sadtalker_paths["mappingnet_checkpoint"] = hf_hub_download(
            repo_id=checkpoint_dir,
            filename="mapping_00229-model.pth.tar",
        )
        sadtalker_paths["facerender_yaml"] = os.path.join(config_dir, "facerender.yaml")

    return sadtalker_paths
