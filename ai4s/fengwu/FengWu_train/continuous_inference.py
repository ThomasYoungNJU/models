import argparse
import copy
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from utils.builder import ConfigBuilder
from utils.logger import get_logger
from utils.accelerator import get_accelerator
import utils.misc as utils


_ACCELERATOR = get_accelerator()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg_params = yaml.load(f, Loader=yaml.FullLoader)
    if "model" not in cfg_params or "dataset" not in cfg_params:
        raise ValueError(f"Invalid config file: {cfg_path}")
    return cfg_params


def configure_constants_len(cfg_params, split):
    dataset_params = cfg_params.get("dataset", {})
    split_params = dataset_params.get("train") or dataset_params.get(split) or {}
    vnames = split_params.get("vnames", None)
    constants_len = len(vnames.get("constants", [])) if vnames is not None else 0
    cfg_params.setdefault("model", {}).setdefault("params", {})["constants_len"] = constants_len
    return constants_len


def resolve_checkpoint(args, cfg_params):
    if args.checkpoint is not None:
        return args.checkpoint

    if args.run_dir is not None:
        checkpoint_path = os.path.join(args.run_dir, args.checkpoint_name)
        if os.path.exists(checkpoint_path):
            return checkpoint_path

    extra_params = cfg_params.get("model", {}).get("params", {}).get("extra_params", {})
    checkpoint_path = extra_params.get("checkpoint_path", None)
    if isinstance(checkpoint_path, str) and os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_checkpoint(model, checkpoint_path, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint_model = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not isinstance(checkpoint_model, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    has_named_submodels = (
        len(checkpoint_model) > 0
        and all(isinstance(value, dict) for value in checkpoint_model.values())
    )

    load_report = {}
    for model_key, sub_model in model.model.items():
        if has_named_submodels:
            checkpoint_key = model_key if model_key in checkpoint_model else next(iter(checkpoint_model))
            state_dict = checkpoint_model[checkpoint_key]
        else:
            checkpoint_key = "<root>"
            state_dict = checkpoint_model

        state_dict = strip_module_prefix(state_dict)
        missing, unexpected = sub_model.load_state_dict(state_dict, strict=strict)
        load_report[model_key] = {
            "checkpoint_key": checkpoint_key,
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
    return load_report


def build_dataset(cfg_params, split, input_len, data_dir=None, process_num=None):
    dataset_cfg = copy.deepcopy(cfg_params.get("dataset", {}))
    if split not in dataset_cfg:
        raise ValueError(f"Split '{split}' not found in dataset config")

    dataset_cfg[split]["length"] = input_len
    if data_dir is not None:
        dataset_cfg[split]["data_dir"] = data_dir
    if process_num is not None:
        dataset_cfg[split]["process_num"] = process_num

    builder = ConfigBuilder(**cfg_params)
    return builder.get_dataset(dataset_params=dataset_cfg, split=split)


def unpack_sample(sample):
    if isinstance(sample, tuple):
        seq = sample[0]
    else:
        seq = sample

    if isinstance(seq, torch.Tensor):
        if seq.ndim == 4:
            seq = [seq[i] for i in range(seq.shape[0])]
        elif seq.ndim == 3:
            seq = [seq]
        else:
            raise ValueError(f"Unsupported tensor sample shape: {tuple(seq.shape)}")

    frames = []
    for frame in seq:
        frame_tensor = torch.as_tensor(frame)
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        elif frame_tensor.ndim != 4:
            raise ValueError(f"Unsupported frame shape: {tuple(frame_tensor.shape)}")
        frames.append(frame_tensor)
    return frames


def get_single_submodel(model):
    if len(model.model) != 1:
        raise NotImplementedError("continuous_inference.py currently supports one sub-model.")
    model_key = next(iter(model.model))
    return model_key, model.model[model_key]


def split_prediction(prediction, frame_channels, constants_len):
    dynamic_channels = frame_channels - constants_len
    pred_channels = prediction.shape[1]

    if pred_channels == dynamic_channels:
        dynamic_prediction = prediction
    elif pred_channels == dynamic_channels * 2:
        dynamic_prediction = prediction[:, :dynamic_channels]
    elif pred_channels == frame_channels:
        dynamic_prediction = prediction[:, constants_len:] if constants_len > 0 else prediction
    elif pred_channels == frame_channels * 2:
        state_prediction = prediction[:, :frame_channels]
        dynamic_prediction = state_prediction[:, constants_len:] if constants_len > 0 else state_prediction
    elif pred_channels > dynamic_channels:
        dynamic_prediction = prediction[:, :dynamic_channels]
    else:
        raise ValueError(
            f"Prediction has {pred_channels} channels, expected {dynamic_channels} "
            f"or {dynamic_channels * 2}."
        )
    return dynamic_prediction


def make_next_frame(previous_frame, dynamic_prediction, constants_len):
    if constants_len == 0:
        return dynamic_prediction
    constants = previous_frame[:, :constants_len]
    return torch.cat((constants, dynamic_prediction), dim=1)


def save_prediction(
    output_path,
    dynamic_prediction,
    full_state,
    constants_len,
    mean,
    std,
    denormalize=False,
    save_full_state=False,
    save_fp16=False,
):
    dynamic_cpu = dynamic_prediction.detach().float().cpu()
    if denormalize:
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
        if dynamic_cpu.shape[1] != mean.shape[1]:
            raise ValueError(
                f"Cannot denormalize {dynamic_cpu.shape[1]} channels with "
                f"{mean.shape[1]} mean/std channels."
            )
        dynamic_cpu = dynamic_cpu * std + mean

    if save_full_state and constants_len > 0:
        constants_cpu = full_state[:, :constants_len].detach().float().cpu()
        save_tensor = torch.cat((constants_cpu, dynamic_cpu), dim=1)
    elif save_full_state:
        save_tensor = dynamic_cpu
    else:
        save_tensor = dynamic_cpu

    array = save_tensor.squeeze(0).numpy()
    if save_fp16:
        array = array.astype(np.float16)
    np.save(output_path, array)


@torch.no_grad()
def run_autoregressive_inference(args, cfg_params, logger):
    constants_len = configure_constants_len(cfg_params, args.split)
    checkpoint_path = resolve_checkpoint(args, cfg_params)

    # The inference checkpoint is loaded explicitly below. Avoid implicit
    # constructor-time loading from training configs so the selected checkpoint
    # is unambiguous.
    cfg_params = copy.deepcopy(cfg_params)
    cfg_params.setdefault("model", {}).setdefault("params", {}).setdefault("extra_params", {})[
        "checkpoint_path"
    ] = None
    cfg_params["logger"] = logger

    if args.cuda >= 0:
        _ACCELERATOR.set_device(args.cuda)

    utils.setup_seed(args.seed)
    logger.info("Building model ...")
    builder = ConfigBuilder(**cfg_params)
    model = builder.get_model()
    model_key, sub_model = get_single_submodel(model)

    if checkpoint_path is None:
        logger.warning("No checkpoint found; running with the model initialization from config.")
    else:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        load_report = load_model_checkpoint(model, checkpoint_path, strict=not args.checkpoint_not_strict)
        for key, report in load_report.items():
            logger.info(
                f"Loaded {key} from {report['checkpoint_key']} "
                f"(missing={len(report['missing'])}, unexpected={len(report['unexpected'])})"
            )

    for module in model.model.values():
        module.eval()

    logger.info("Building dataset ...")
    dataset = build_dataset(
        cfg_params,
        split=args.split,
        input_len=args.input_len,
        data_dir=args.data_dir,
        process_num=args.process_num,
    )
    mean, std = dataset.get_meanstd()
    mean = mean.float()
    std = std.float()

    saved_steps = list(range(args.save_interval, args.pred_len + 1, args.save_interval))
    if args.num_samples < 0:
        sample_indices = list(range(args.sample_start, len(dataset), args.sample_interval))
    else:
        sample_indices = [
            args.sample_start + sample_offset * args.sample_interval
            for sample_offset in range(args.num_samples)
        ]
    if args.shard_count > 1:
        sample_indices = [
            sample_index
            for offset, sample_index in enumerate(sample_indices)
            if offset % args.shard_count == args.shard_id
        ]
    sample_indices = [sample_index for sample_index in sample_indices if sample_index < len(dataset)]
    if len(sample_indices) == 0:
        raise ValueError("No samples selected for inference")
    metadata = {
        "cfg": args.cfg,
        "checkpoint": checkpoint_path,
        "split": args.split,
        "sample_start": args.sample_start,
        "num_samples": args.num_samples,
        "sample_interval": args.sample_interval,
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
        "sample_indices": sample_indices,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "save_interval": args.save_interval,
        "saved_steps": saved_steps,
        "model_key": model_key,
        "constants_len": constants_len,
        "denormalize": args.denormalize,
        "save_full_state": args.save_full_state,
        "save_fp16": args.save_fp16,
    }
    with open(os.path.join(args.output_dir, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f, sort_keys=False)
    np.savetxt(
        os.path.join(args.output_dir, "sample_indices.txt"),
        np.asarray(sample_indices, dtype=np.int64),
        fmt="%d",
    )

    for sample_index in sample_indices:
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_index:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        logger.info(f"Running sample index {sample_index} ...")
        frames = unpack_sample(dataset[sample_index])
        #print("frames", len(frames), args.input_len)
        if len(frames) < args.input_len:
            raise ValueError(f"Sample has {len(frames)} frames, needs input_len={args.input_len}")

        state = [
            frame.float().to(model.device, non_blocking=True)
            for frame in frames[: args.input_len]
        ]
        frame_channels = state[-1].shape[1]

        for lead_step in range(1, args.pred_len + 1):
            #print("state", len(state))
            model_input = torch.cat(state[-args.input_len :], dim=1)
            #print("input", model_input.shape)
            prediction = sub_model(model_input)
            dynamic_prediction = split_prediction(
                prediction,
                frame_channels=frame_channels,
                constants_len=constants_len,
            )
            next_frame = make_next_frame(state[-1], dynamic_prediction, constants_len)

            if lead_step % args.save_interval == 0:
                output_path = os.path.join(sample_dir, f"pred_step_{lead_step:03d}.npy")
                save_prediction(
                    output_path,
                    dynamic_prediction,
                    next_frame,
                    constants_len=constants_len,
                    mean=mean,
                    std=std,
                    denormalize=args.denormalize,
                    save_full_state=args.save_full_state,
                    save_fp16=args.save_fp16,
                )
                logger.info(f"Saved step {lead_step}: {output_path}")

            state = state[1:] + [next_frame.detach()]

        del state
        _ACCELERATOR.empty_cache()

    logger.info("Continuous inference complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./config/fengwu_single_sample.yaml")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory used to find checkpoint_latest.pth when --checkpoint is not set.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_latest.pth")
    parser.add_argument("--output_dir", type=str, default="./continuous_inference_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of selected samples to run. Use -1 to scan to the end of the split.",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=4,
        help="Run one sample every N dataset indices. Default keeps sample ids 0, 4, 8, ...",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Zero-based shard id after sample_interval selection.",
    )
    parser.add_argument(
        "--shard_count",
        type=int,
        default=1,
        help="Number of shards after sample_interval selection.",
    )
    parser.add_argument("--input_len", type=int, default=2)
    parser.add_argument("--pred_len", type=int, default=60)
    parser.add_argument("--save_interval", type=int, default=4)
    parser.add_argument("--process_num", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denormalize", action="store_true")
    parser.add_argument("--save_full_state", action="store_true")
    parser.add_argument("--save_fp16", action="store_true")
    parser.add_argument("--checkpoint_not_strict", action="store_true")
    args = parser.parse_args()

    if args.input_len < 1:
        raise ValueError("--input_len must be positive")
    if args.pred_len < 1:
        raise ValueError("--pred_len must be positive")
    if args.save_interval < 1:
        raise ValueError("--save_interval must be positive")
    if args.sample_interval < 1:
        raise ValueError("--sample_interval must be positive")
    if args.num_samples == 0:
        raise ValueError("--num_samples must be positive, or -1 to scan to the end")
    if args.shard_count < 1:
        raise ValueError("--shard_count must be positive")
    if args.shard_id < 0 or args.shard_id >= args.shard_count:
        raise ValueError("--shard_id must be in [0, shard_count)")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("continuous_inference", args.output_dir, 0, filename="inference.log")

    cfg_params = load_config(args.cfg)
    run_autoregressive_inference(args, cfg_params, logger)


if __name__ == "__main__":
    main()
