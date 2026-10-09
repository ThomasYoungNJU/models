import time
_PROGRAM_START = time.perf_counter()

import argparse
import os
import torch
from utils.accelerator import get_accelerator, configure_fp32_acceleration, format_fp32_acceleration_status
from utils.flaggems_runtime import configure_flag_gems, format_flag_gems_status

_ACCELERATOR = get_accelerator()
_FP32_ACCEL_STATUS = configure_fp32_acceleration(_ACCELERATOR)
_FLAGGEMS_STATUS = configure_flag_gems(_ACCELERATOR, stage="early")

from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger


_TIMING_ENABLED = os.environ.get("FENGWU_TIMING", "0").lower() not in {"0", "false", "off", "no"}
_RUN_TIMING_SUMMARY = {}


def _elapsed_host(start):
    """Host wall time to Python call return; never forces CUDA completion."""
    return time.perf_counter() - start


def _record_accelerator_event():
    """Record a non-blocking marker on the current accelerator stream."""
    if not _TIMING_ENABLED:
        return None
    return _ACCELERATOR.record_event()


def _accelerator_span_seconds(start_event, end_event):
    """Resolve only completed events.  Never synchronize merely for timing."""
    return _ACCELERATOR.event_elapsed_seconds(start_event, end_event)


#----------------------------------------------------------------------------

def subprocess_fn(args):
    global _FLAGGEMS_STATUS

    startup_components = {}
    phase_start = time.perf_counter()
    utils.setup_seed(args.seed * args.world_size + args.rank)
    if args.deterministic:
        utils.setup_deterministic()

    logger = get_logger("train", args.run_dir, utils.get_rank(), filename='iter.log', resume=args.resume)
    seed_elapsed = _elapsed_host(phase_start)
    startup_components["seed_deterministic_logger"] = seed_elapsed
    logger.info("[Runtime] {}".format(_ACCELERATOR.summary()))
    logger.info("[FP32Accel] {}".format(format_fp32_acceleration_status(_FP32_ACCEL_STATUS)))
    logger.info("[FlagGems] {}".format(format_flag_gems_status(_FLAGGEMS_STATUS)))
    logger.info("[Timing][startup host] seed_deterministic_logger: {:.6f}s".format(seed_elapsed))

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    phase_start = time.perf_counter()
    builder = ConfigBuilder(**args.cfg_params)
    config_builder_elapsed = _elapsed_host(phase_start)
    startup_components["config_builder"] = config_builder_elapsed
    logger.info("[Timing][startup host] config_builder: {:.6f}s".format(config_builder_elapsed))

    logger.info('Building dataloaders ...')

    phase_start = time.perf_counter()
    train_dataloader = builder.get_dataloader(split = 'train')
    train_dataloader_elapsed = _elapsed_host(phase_start)
    startup_components["train_dataloader"] = train_dataloader_elapsed
    logger.info('Train dataloaders build complete')
    logger.info("[Timing][startup host] train_dataloader: {:.6f}s (steps={})".format(
        train_dataloader_elapsed, len(train_dataloader)))
    phase_start = time.perf_counter()
    test_dataloader = builder.get_dataloader(split = 'test')
    test_dataloader_elapsed = _elapsed_host(phase_start)
    startup_components["test_dataloader"] = test_dataloader_elapsed
    logger.info('Test dataloaders build complete')
    logger.info("[Timing][startup host] test_dataloader: {:.6f}s (steps={})".format(
        test_dataloader_elapsed, len(test_dataloader)))

    model_params = args.cfg_params['model']['params']
    if 'train_steps' in model_params:
        steps_per_epoch = model_params['train_steps']
    else:
        steps_per_epoch = len(train_dataloader)


    # steps_per_epoch = len(train_dataloader)

    if 'lr_scheduler' in model_params:
        lr_scheduler_params = model_params['lr_scheduler']
        for key in lr_scheduler_params:
            if 'by_step' in lr_scheduler_params[key]:
                if lr_scheduler_params[key]['by_step']:
                    for key1 in lr_scheduler_params[key]:
                        if "epochs" in key1:
                            lr_scheduler_params[key][key1] *= steps_per_epoch

    # Host timings below end at Python call return.  Accelerator events
    # are resolved later, after the existing training loss.item() synchronization.
    logger.info('Building models ...')
    model_accelerator_start = _record_accelerator_event()
    phase_start = time.perf_counter()
    model = builder.get_model()
    model_host_elapsed = _elapsed_host(phase_start)
    startup_components["model_optimizer_scheduler_build_call"] = model_host_elapsed
    model_accelerator_end = _record_accelerator_event()
    logger.info("[Timing][startup host] model_optimizer_scheduler_build_call: {:.6f}s".format(
        model_host_elapsed))

    model_checkpoint = os.path.join(args.run_dir, 'checkpoint_latest.pth')
    resume_accelerator_start = _record_accelerator_event()
    phase_start = time.perf_counter()
    if args.resume:
        model.load_checkpoint(model_checkpoint, resume=True)
    resume_host_elapsed = _elapsed_host(phase_start)
    startup_components["resume_checkpoint_load_call"] = resume_host_elapsed
    resume_accelerator_end = _record_accelerator_event()
    logger.info("[Timing][startup host] resume_checkpoint_load_call: {:.6f}s".format(
        resume_host_elapsed))

    for key in model.model:
        params = [p for p in model.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])


        logger.info("params {key}: {cnt_params}".format(key=key, cnt_params=cnt_params))

    if args.init_only:
        logger.info('init_only enabled, saving initial checkpoint and exiting ...')
        model.save_checkpoint(-1, args.run_dir, save_type='save_latest')
        return

    _FLAGGEMS_STATUS = configure_flag_gems(_ACCELERATOR, stage="training")
    logger.info("[FlagGems] training stage: {}".format(
        format_flag_gems_status(_FLAGGEMS_STATUS)))

    fp32_current = configure_fp32_acceleration(_ACCELERATOR)
    logger.info("[FP32Accel] training stage: {}".format(format_fp32_acceleration_status(fp32_current)))
    logger.info("[TimingScope] host-call spans; device Events include stream idle/launch gaps, not pure kernel time")

    startup_to_training = _elapsed_host(_PROGRAM_START)
    logger.info("[Timing][startup host] program_start_to_training: {:.6f}s".format(
        startup_to_training))
    logger.info('begin training ...')

    phase_start = time.perf_counter()
    trainer_summary = model.trainer(
        train_dataloader, test_dataloader, builder.get_max_epoch(),
        checkpoint_savedir=args.run_dir, resume=args.resume)
    trainer_call_total = _elapsed_host(phase_start)
    logger.info("[Timing][host] trainer_call_total: {:.6f}s".format(trainer_call_total))

    if _TIMING_ENABLED:
        for name, start_event, end_event in (
            (
                "model_optimizer_scheduler_build_stream_span",
                model_accelerator_start,
                model_accelerator_end,
            ),
            (
                "resume_checkpoint_load_stream_span",
                resume_accelerator_start,
                resume_accelerator_end,
            ),
        ):
            elapsed = _accelerator_span_seconds(start_event, end_event)
            if elapsed is None:
                logger.info(
                    "[Timing][startup accelerator] {}: pending "
                    "(not synchronized for timing)".format(name))
            else:
                logger.info(
                    "[Timing][startup accelerator] {}: {:.6f}s".format(
                        name, elapsed))

        startup_accounted = sum(startup_components.values())
        startup_other = max(0.0, startup_to_training - startup_accounted)
        startup_parts = [
            "{}={:.6f}s".format(name, value)
            for name, value in startup_components.items()
        ]
        logger.info(
            "[Timing][SUMMARY][startup host components] {} accounted={:.6f}s "
            "other_imports_args_io={:.6f}s startup_to_training={:.6f}s".format(
                " ".join(startup_parts), startup_accounted, startup_other,
                startup_to_training))

        program_to_summary = _elapsed_host(_PROGRAM_START)
        after_trainer = max(0.0, program_to_summary - startup_to_training - trainer_call_total)
        logger.info(
            "[Timing][SUMMARY][run major blocks] startup_to_training={:.6f}s "
            "trainer_call={:.6f}s after_trainer_summary={:.6f}s "
            "program_to_summary={:.6f}s train_steps={}".format(
                startup_to_training, trainer_call_total, after_trainer,
                program_to_summary, trainer_summary.get("train_step_count", 0)))
        _RUN_TIMING_SUMMARY.update({
            "startup_to_training": startup_to_training,
            "trainer_call": trainer_call_total,
            "program_to_summary": program_to_summary,
        })


def main(args):
    print(args.desc)
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = 0
        _ACCELERATOR.set_device(args.local_rank)
    desc = f'world_size{args.world_size:d}'

    if args.desc is not None:
        desc += f'-{args.desc}'

    alg_dir = args.cfg.split("/")[-1].split(".")[0]
    alg_dir_head = args.cfg.split("/")[-2]
    # args.outdir = args.outdir + "/" + alg_dir_head + "/" + alg_dir
    args.outdir = args.outdir + "/" + alg_dir
    run_dir = os.path.join(args.outdir, f'{desc}')
    relative_checkpoint_dir = alg_dir + "/" + f'{desc}'
    args.relative_checkpoint_dir = relative_checkpoint_dir
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    train_config_file = os.path.join(run_dir, 'training_options.yaml')

    if (not args.resume) or args.resume_from_config or (not os.path.exists(train_config_file)):
        print("load yaml from config")
        with open(args.cfg, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)


    else:
        print("load yaml from resume")
        with open(train_config_file, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        del_keys = []
        for key in cfg_params:
            if key in args:
                del_keys.append(key)
        for key in del_keys:
            del cfg_params[key]

    # cfg_params['dataloader']['num_workers'] = args.per_cpus
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)
    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get('constants'))
    else:
        constants_len = 0
    cfg_params['model']['params']['constants_len'] = constants_len

    if args.rank == 0:
        with open(os.path.join(run_dir, 'training_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    print('Launching processes...')
    subprocess_fn(args)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size',     type = int,     default = 1,                                            help = 'tensor_model_parallel_size')
    parser.add_argument('--pipeline_model_parallel_size',   type = int,     default = 1,                                            help = 'pipeline_model_parallel_size')
    parser.add_argument('--resume',                         action = "store_true",                                                  help = 'resume')
    parser.add_argument('--resume_from_config',             action = "store_true",                                                  help = 'resume from config')
    parser.add_argument('--seed',                           type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',                           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',                     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',                       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--init_method',                    type = str,     default='tcp://127.0.0.1:23456',                        help = 'multi process init method')
    parser.add_argument('--outdir',                         type = str,     default='./IF_WKP',  help = 'Where to save the results')
    parser.add_argument('--cfg', '-c',                      type = str,     default = os.path.join('configs', 'default.yaml'),      help = 'path to the configuration file')
    parser.add_argument('--desc',                           type=str,       default='STR',                                          help = 'String to include in result dir name')
    parser.add_argument('--deterministic',                  action = "store_true",                                                  help = 'enable deterministic runtime settings where possible')
    parser.add_argument('--init_only',                      action = "store_true",                                                  help = 'initialize model, save checkpoint_latest.pth, and exit')


    args = parser.parse_args()

    try:
        main(args)
    finally:
        # No timing-only accelerator synchronization here.  Successful training has
        # already reached its original loss.item()/checkpoint completion points.
        program_total = time.perf_counter() - _PROGRAM_START
        print("[Timing][host] TOTAL (Python entry to final timing point): {:.6f}s".format(
            program_total), flush=True)
        if _TIMING_ENABLED and _RUN_TIMING_SUMMARY:
            after_summary = max(
                0.0, program_total - _RUN_TIMING_SUMMARY["program_to_summary"])
            print(
                "[Timing][SUMMARY][program total] total={:.6f}s "
                "startup_to_training={:.6f}s trainer_call={:.6f}s "
                "summary_and_exit={:.6f}s".format(
                    program_total,
                    _RUN_TIMING_SUMMARY["startup_to_training"],
                    _RUN_TIMING_SUMMARY["trainer_call"],
                    after_summary),
                flush=True)
