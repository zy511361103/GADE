import os
import torch
from torch import nn
from leaf_pytorch.frontend import Leaf
from utils import str_to_bool


def get_frontend(opt):
    front_end_config = opt["frontend"]
    audio_config = opt["audio_config"]

    pretrained = front_end_config.get("pretrained", "")
    if os.path.isfile(pretrained):
        pretrained_flag = True
        ckpt = torch.load(pretrained)
    else:
        pretrained_flag = False

    if "leaf" in front_end_config['name'].lower():
        default_args = str_to_bool(front_end_config["default_args"])
        use_legacy_complex = str_to_bool(front_end_config["use_legacy_complex"])
        initializer = str(front_end_config["initializer"])
        if default_args:
            print("Using default Leaf arguments..")
            fe = Leaf(use_legacy_complex=use_legacy_complex, initializer=initializer)
        else:
            sr = int(audio_config["sample_rate"])
            window_len_ms = float(audio_config["window_len"])
            window_stride_ms = float(audio_config["window_stride"])  # 10换1,决定最后一维的大小
            n_filters = int(front_end_config["n_filters"])
            min_freq = float(front_end_config["min_freq"])
            max_freq = float(front_end_config["max_freq"])
            pcen_compress = str_to_bool(front_end_config["pcen_compress"])
            mean_var_norm = str_to_bool(front_end_config["mean_var_norm"])
            preemp = str_to_bool(front_end_config["preemp"])
            fe = Leaf(
                n_filters=n_filters,
                sample_rate=sr,
                window_len=window_len_ms,
                window_stride=window_stride_ms,
                preemp=preemp,
                init_min_freq=min_freq,
                init_max_freq=max_freq,
                mean_var_norm=mean_var_norm,
                pcen_compression=pcen_compress,
                use_legacy_complex=use_legacy_complex,
                initializer=initializer
            )
    else:
        raise NotImplementedError("Other front ends not implemented yet.")
    if pretrained_flag:
        print("attempting to load pretrained frontend weights..", fe.load_state_dict(ckpt))
    return fe
