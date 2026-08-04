import torch
import json
import torch.nn as nn
from networks.LGUnet_all import LGUnet_all
from utils.builder import get_optimizer, get_lr_scheduler
import utils.misc as utils
import time
import datetime
from pathlib import Path
import os
from collections import OrderedDict
from torch.functional import F
from utils.misc import is_dist_avail_and_initialized
from replay.replay_buff import replay_buff
import gc
import math
import statistics
from utils.accelerator import get_accelerator


_ACCELERATOR = get_accelerator()

LOG_SIG_MAX = 5
LOG_SIG_MIN = -8


def _timing_values(records, key):
    """Return finite numeric values already observed by the training path."""
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def _timing_stats(values):
    """Population statistics for the complete set of steps in this run."""
    if not values:
        return None
    return {
        "count": len(values),
        "sum": sum(values),
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger
        self.timing_enabled = os.environ.get("FENGWU_TIMING", "0").lower() not in {"0", "false", "off", "no"}
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None
        self.constants_len = self.params.get("constants_len", 0)
        self.extra_params = params.get("extra_params", {})
        self.loss_type = self.extra_params.get("loss_type", "PossLoss")
        self.whether_save_checkpoint = self.extra_params.get("whether_save_checkpoint", True)
        self.save_best = self.extra_params.get("save_best", True)
        self.save_last = self.extra_params.get("save_last", True)
        self.freeze_parameters = self.extra_params.get("freeze_parameters", {})

        self.replay_buff_params = self.extra_params.get("replay_buff", None)
        # if self.two_step_training:
        self.checkpoint_path = self.extra_params.get('checkpoint_path', None)
        self.checkpoint_strict = self.extra_params.get("checkpoint_strict", True)


        checkpoint_dir = self.extra_params.get("checkpoint_dir", "weatherbench:s3://weatherbench/checkpoint")
        self.save_checkpoint_dir = self.extra_params.get('save_checkpoint_dir', checkpoint_dir)

        self.begin_epoch = 0
        self.metric_best = 1000

        device_index = utils.get_localrank() if is_dist_avail_and_initialized() else 0
        device = _ACCELERATOR.device(device_index)
        if is_dist_avail_and_initialized() and device.type == "cpu":
            raise EnvironmentError(
                "No accelerator is available; cannot initialize multidevice training.")

        self.device = device
        sub_model = params.get('sub_model', {})
        for key in sub_model:
            origin_key = key
            if key[-5:] == "_copy":
                key = key[:-5]
            if key == "lgunet_all":
                model = LGUnet_all(**sub_model[origin_key])
            else:
                raise NotImplementedError('Invalid model type.')
            self.model[origin_key] = model
            key = origin_key
            if self.loss_type == "Possloss":
                output_dim = self.params['sub_model'][list(self.model.keys())[0]]["out_chans"]
                img_size = self.params['sub_model'][list(self.model.keys())[0]].get("img_size", [32, 64])

                self.max_logvar = self.model[key].max_logvar = torch.nn.Parameter((torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() / 2))
                self.min_logvar = self.model[key].min_logvar = torch.nn.Parameter((-torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() * 10))


            self.model[key].to(device)
            if is_dist_avail_and_initialized():
                parallel_model = torch.nn.parallel.DistributedDataParallel(
                    self.model[key],
                    device_ids=_ACCELERATOR.ddp_device_ids(utils.get_localrank()),
                )

                self.model[key] = parallel_model

            self.sub_model_name.append(key)

        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})

        for key in self.sub_model_name:
            if key in optimizer:
                self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])

            if key in lr_scheduler:
                self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        # load metrics

        self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()


        if self.checkpoint_path is None:
            self.logger.info("finetune checkpoint path not exist")
        else:
            if isinstance(self.checkpoint_path, str):
                self.load_checkpoint(self.checkpoint_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
            elif isinstance(self.checkpoint_path, dict):
                for key in self.checkpoint_path:
                    if isinstance(self.checkpoint_path[key], str):
                        self.load_checkpoint(self.checkpoint_path[key], load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False, load_parameters=key)
                    else:
                        self.load_checkpoint(key, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False, load_parameters=self.checkpoint_path[key])


        if self.loss_type == "Possloss":
            self.loss = self.Possloss


    def to(self, device):
        self.device = device
        for key in self.model:
            self.model[key].to(device)
        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def data_preprocess(self, data):
        return None, None


    def Possloss(self, pred, target, **kwargs):
        # print(pred.shape, target.shape, self.max_logvar.shape, self.min_logvar.shape)
        inc_var_loss = kwargs.get("inc_var_loss", True)

        num_examples = pred.size()[0]

        mean, log_var = pred.chunk(2, dim = 1)
        # log_var = torch.tanh(log_var)

        # mean = mean.reshape(num_examples, -1)
        log_var = log_var.reshape(num_examples, -1)
        # target = target.reshape(num_examples, -1)

        _, C, H, W = target.shape
        max_logvar = self.max_logvar
        min_logvar = self.min_logvar



        log_var = max_logvar - F.softplus(max_logvar - log_var)
        log_var = min_logvar + F.softplus(log_var - min_logvar)

        log_var = log_var.reshape(*(target.shape))

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var)
            var_loss = torch.mean(log_var)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2))
            total_loss = mse_loss
        total_loss += 0.01 * torch.mean(max_logvar) - 0.01 * torch.mean(min_logvar)


        return torch.mean(total_loss)


    def train_one_step(self, batch_data, step):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)
        else:
            raise NotImplementedError('Invalid model type.')

        loss = self.loss(predict, target)
        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
        else:
            raise NotImplementedError('Invalid model type.')

        return loss


    def test_one_step(self, batch_data):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)

        data_dict = {}
        data_dict['gt'] = target
        data_dict['pred'] = predict
        return data_dict


    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)


        # test_logger = {}


        end_time = time.perf_counter()
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')

        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'

        if train_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = train_data_loader

        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch*max_step+step)

            # The iterator has already produced batch here, so this is data wait/load time.
            data_elapsed = time.perf_counter() - end_time
            data_time.update(data_elapsed)

            loss = self.train_one_step(batch, step)

            metric_logger.update(**loss)
            step_total = time.perf_counter() - end_time
            iter_time.update(step_total)
            timing = getattr(self, "last_step_timing", None)
            if self.timing_enabled:
                timing_record = {"data": data_elapsed, "step_total": step_total}
                if timing:
                    timing_record.update(timing)
                self._train_step_timing_records.append(timing_record)
            if timing:
                self.logger.info(
                    "[Timing][train step {}/{}][host] data={:.6f}s preprocess_enqueue={:.6f}s "
                    "zero_grad_enqueue={:.6f}s forward_enqueue={:.6f}s loss_enqueue={:.6f}s "
                    "backward_enqueue={:.6f}s optimizer_enqueue={:.6f}s natural_sync_wait={:.6f}s "
                    "compute_total={:.6f}s step_total={:.6f}s events_ready={}".format(
                        step + 1, max_step, data_elapsed,
                        timing.get("preprocess_h2d_host_enqueue", 0.0),
                        timing.get("zero_grad_host_enqueue", 0.0),
                        timing.get("forward_host_enqueue", 0.0),
                        timing.get("loss_host_enqueue", 0.0),
                        timing.get("backward_host_enqueue", 0.0),
                        timing.get("optimizer_host_enqueue", 0.0),
                        timing.get("loss_item_natural_sync", 0.0),
                        timing.get("compute_host_total", 0.0), step_total,
                        timing.get("cuda_events_ready", False)))
                self.logger.info(
                    "[Timing][train step {}/{}][cuda event] preprocess_h2d={:.6f}s "
                    "zero_grad={:.6f}s forward={:.6f}s loss={:.6f}s backward={:.6f}s "
                    "optimizer={:.6f}s mode={}".format(
                        step + 1, max_step,
                        timing.get("preprocess_h2d_cuda", float("nan")),
                        timing.get("zero_grad_cuda", float("nan")),
                        timing.get("forward_cuda", float("nan")),
                        timing.get("loss_cuda", float("nan")),
                        timing.get("backward_cuda", float("nan")),
                        timing.get("optimizer_cuda", float("nan")),
                        timing.get("mode", "unknown")))
            end_time = time.perf_counter()

            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                memory_reserved = _ACCELERATOR.memory_reserved() / (1024. * 1024)
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=memory_reserved,
                        meters=str(metric_logger)
                    ))

        return metric_logger


    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=True, load_scheduler=True,
                        load_epoch=True, load_metric_best=True, resume=False, load_parameters=[], checkpoint_strict=None):




        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        else:
            self.logger.info("checkpoint is not exist")
            return
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        if load_model:
            for key in self.model:
                if not isinstance(load_parameters, list):
                    load_parameters = [load_parameters,]
                if len(load_parameters) > 0:
                    if key != load_parameters[0]:
                        continue
                new_state_dict = OrderedDict()
                # model_state_dict = self.model[key]
                if resume:
                    checkpoint_key = key
                    if not (key in checkpoint_model):
                        continue
                else:
                    checkpoint_key = list(checkpoint_model.keys())[0]
                for k, v in checkpoint_model[checkpoint_key].items():
                    if is_dist_avail_and_initialized() and "module" == k[:6]:
                        name = k
                    elif is_dist_avail_and_initialized():
                        name = f"module.{k}"
                    elif "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k

                    if len(load_parameters) > 1  and not resume:
                        for load_parameter in load_parameters[1:]:
                            if load_parameter in name:
                                new_state_dict[name] = v
                                break
                    elif key in self.freeze_parameters and len(self.freeze_parameters[key]) > 0 and not resume:
                        for freeze_parameter in self.freeze_parameters[key]:
                            if freeze_parameter in name:
                                new_state_dict[name] = v
                                break
                    else:
                        new_state_dict[name] = v

                with torch.no_grad():
                    self.model[key].load_state_dict(new_state_dict, strict=self.checkpoint_strict if checkpoint_strict is None else checkpoint_strict)
                self.model[key].to(self.device)
                del new_state_dict

        if load_optimizer:
            for key in checkpoint_optimizer:
                self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
        if load_scheduler:
            for key in checkpoint_lr_scheduler:
                self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        if load_epoch:
            self.begin_epoch = checkpoint_dict['epoch']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']




        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=checkpoint_dict['epoch'], metric_best=checkpoint_dict['metric_best'] if 'metric_best' in checkpoint_dict else None))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'):
        checkpoint_savedir = Path(checkpoint_savedir)

        if utils.get_rank() == 0:
            if save_type == "save_best":
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth')
            else:
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest.pth')


        if utils.get_world_size() > 1 and utils.get_rank() == 0:

            torch.save(
                {
                'epoch':            epoch+1,
                'model':            {key: self.model[key].module.state_dict() for key in self.model},
                'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                'metric_best':      self.metric_best,
                # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                }, checkpoint_path
            )
        elif utils.get_world_size() == 1:

            torch.save(
                {
                'epoch':            epoch+1,
                'model':            {key: self.model[key].state_dict() for key in self.model},
                'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                'metric_best':      self.metric_best,
                # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                }, checkpoint_path
            )


    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, max_epoches, checkpoint_savedir=None, save_ceph=False, resume=False):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        if self.replay_buff_params is not None:
            self.replay_buff = replay_buff(train_data_loader, **(self.replay_buff_params))
        if train_data_loader is not None:
            data_std = train_data_loader.dataset.get_meanstd()[1]
            if type(data_std) == torch.Tensor:
                data_std = train_data_loader.dataset.get_meanstd()[1].float()
            else:
                data_std = torch.Tensor(train_data_loader.dataset.get_meanstd()[1]).float()
        else:
            data_std = None

        if data_std.shape[-1] == 1:
            data_std = data_std.squeeze(-1).squeeze(-1)
        self.datastd = data_std.to(self.device)


        if utils.get_world_size() > 1:
            for key in self.model:
                utils.check_ddp_consistency(self.model[key])
        self.now_step = self.begin_epoch * len(train_data_loader)
        loss_log_path = Path(checkpoint_savedir) / "training_loss.jsonl"
        loss_log = loss_log_path.open("w", encoding="utf-8")
        trainer_start = time.perf_counter()
        train_compute_total = 0.0
        post_training_total = 0.0
        validation_total = 0.0
        loss_log_total = 0.0
        checkpoint_total = 0.0
        cleanup_total = 0.0
        self._train_step_timing_records = []
        for epoch in range(self.begin_epoch, max_epoches):
            if train_data_loader is not None:
                train_data_loader.sampler.set_epoch(epoch)


            phase_start = time.perf_counter()
            train_metric_logger = self.train_one_epoch(train_data_loader, epoch, max_epoches)
            # train_one_epoch ends after the original per-step loss.item().
            train_elapsed = time.perf_counter() - phase_start
            train_compute_total += train_elapsed

            post_epoch_start = time.perf_counter()
            # # update lr_scheduler
            if utils.get_world_size() > 1:
                for key in self.model:
                    utils.check_ddp_consistency(self.model[key])

            phase_start = time.perf_counter()
            metric_logger = self.test(test_data_loader, epoch)
            # test_one_step also ends at its original loss.item().
            validation_elapsed = time.perf_counter() - phase_start
            validation_total += validation_elapsed

            phase_start = time.perf_counter()
            entry = {"step": epoch + 1}
            if self.loss_type in train_metric_logger.meters:
                entry["train_loss"] = round(train_metric_logger.meters[self.loss_type].global_avg, 6)
            if self.loss_type in metric_logger.meters:
                entry["val_loss"] = round(metric_logger.meters[self.loss_type].global_avg, 6)
            loss_log.write(json.dumps(entry) + "\n")
            loss_log.flush()
            loss_log_elapsed = time.perf_counter() - phase_start
            loss_log_total += loss_log_elapsed

            phase_start = time.perf_counter()
            if self.whether_save_checkpoint:
                if self.save_best and self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                if self.save_last and (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')
            checkpoint_elapsed = time.perf_counter() - phase_start
            checkpoint_total += checkpoint_elapsed

            phase_start = time.perf_counter()
            gc.collect()
            if is_dist_avail_and_initialized():
                torch.distributed.barrier()
            cleanup_elapsed = time.perf_counter() - phase_start
            cleanup_total += cleanup_elapsed

            post_epoch_elapsed = time.perf_counter() - post_epoch_start
            post_training_total += post_epoch_elapsed
            self.logger.info(
                "[Timing][epoch {}] train={:.6f}s validation={:.6f}s loss_log={:.6f}s "
                "checkpoint={:.6f}s gc_barrier={:.6f}s post_training={:.6f}s".format(
                    epoch + 1, train_elapsed, validation_elapsed, loss_log_elapsed,
                    checkpoint_elapsed, cleanup_elapsed, post_epoch_elapsed))

        phase_start = time.perf_counter()
        loss_log.close()
        loss_log_close_elapsed = time.perf_counter() - phase_start
        post_training_total += loss_log_close_elapsed
        trainer_elapsed = time.perf_counter() - trainer_start
        self.logger.info("Structured loss log saved to: {}".format(loss_log_path))
        self.logger.info(
            "[Timing][trainer summary] train_compute={:.6f}s post_training={:.6f}s "
            "loss_log_close={:.6f}s trainer_total={:.6f}s".format(
                train_compute_total, post_training_total, loss_log_close_elapsed, trainer_elapsed))

        accounted_total = (
            train_compute_total + validation_total + loss_log_total +
            checkpoint_total + cleanup_total + loss_log_close_elapsed
        )
        trainer_other = max(0.0, trainer_elapsed - accounted_total)
        self.logger.info(
            "[Timing][SUMMARY][trainer blocks] epochs={} train={:.6f}s validation={:.6f}s "
            "loss_log={:.6f}s checkpoint={:.6f}s gc_barrier={:.6f}s "
            "loss_log_close={:.6f}s other={:.6f}s trainer_total={:.6f}s".format(
                max_epoches - self.begin_epoch, train_compute_total, validation_total,
                loss_log_total, checkpoint_total, cleanup_total, loss_log_close_elapsed,
                trainer_other, trainer_elapsed))

        if self.timing_enabled:
            records = self._train_step_timing_records
            step_stats = _timing_stats(_timing_values(records, "step_total"))
            if step_stats is not None:
                self.logger.info(
                    "[Timing][SUMMARY][train step total] count={count} sum={sum:.6f}s "
                    "avg={avg:.6f}s median={median:.6f}s std_population={std:.6f}s "
                    "min={min:.6f}s max={max:.6f}s".format(**step_stats))

            host_components = (
                ("data", "data"),
                ("preprocess_enqueue", "preprocess_h2d_host_enqueue"),
                ("zero_grad_enqueue", "zero_grad_host_enqueue"),
                ("forward_enqueue", "forward_host_enqueue"),
                ("loss_enqueue", "loss_host_enqueue"),
                ("backward_enqueue", "backward_host_enqueue"),
                ("optimizer_enqueue", "optimizer_host_enqueue"),
                ("natural_sync_wait", "loss_item_natural_sync"),
                ("compute_total", "compute_host_total"),
            )
            host_parts = []
            for label, key in host_components:
                stats = _timing_stats(_timing_values(records, key))
                if stats is not None:
                    host_parts.append(
                        "{}_sum={:.6f}s(avg={:.6f}s)".format(label, stats["sum"], stats["avg"]))
            self.logger.info("[Timing][SUMMARY][train step host component sums] {}".format(
                " ".join(host_parts) if host_parts else "unavailable"))

            cuda_components = (
                ("preprocess_h2d", "preprocess_h2d_cuda"),
                ("zero_grad", "zero_grad_cuda"),
                ("forward", "forward_cuda"),
                ("loss", "loss_cuda"),
                ("backward", "backward_cuda"),
                ("optimizer", "optimizer_cuda"),
            )
            cuda_parts = []
            for label, key in cuda_components:
                stats = _timing_stats(_timing_values(records, key))
                if stats is not None:
                    cuda_parts.append(
                        "{}_sum={:.6f}s(avg={:.6f}s,n={})".format(
                            label, stats["sum"], stats["avg"], stats["count"]))
            self.logger.info(
                "[Timing][SUMMARY][train step accelerator component sums] {}".format(
                " ".join(cuda_parts) if cuda_parts else "unavailable"))

        return {
            "train": train_compute_total,
            "validation": validation_total,
            "loss_log": loss_log_total,
            "checkpoint": checkpoint_total,
            "gc_barrier": cleanup_total,
            "loss_log_close": loss_log_close_elapsed,
            "other": trainer_other,
            "trainer_total": trainer_elapsed,
            "train_step_count": len(self._train_step_timing_records),
        }

    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # set model to eval
        for key in self.model:
            self.model[key].eval()


        max_step = len(test_data_loader)

        if test_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = test_data_loader


        # max_step = len(iter(test_data_loader))
        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None

            loss = self.test_one_step(batch)
            metric_logger.update(**loss)

        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger
