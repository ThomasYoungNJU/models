import os
import time
import torch
from models.model import basemodel
from utils.accelerator import get_accelerator


_ACCELERATOR = get_accelerator()


class MTS2d_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)

    def data_preprocess(self, data):
        # print(len(data))
        # begin_time = time.time()
        if isinstance(data[0], list):
            data = data[0]
        inp = [data[0]]
        for i in range(1, len(data)-1):
            inp.append(data[i])
        inp = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)

        tar_step1 = data[-1].float().to(self.device, non_blocking=True)

        return inp, tar_step1



    def train_one_step(self, batch_data, step):
        timing_enabled = getattr(self, "timing_enabled", False)

        # Exact original fast path: no accelerator events, timing calls, or timing logs.
        if not timing_enabled:
            inp, tar_step1 = self.data_preprocess(batch_data)
            optimizer = self.optimizer[list(self.model.keys())[0]]
            optimizer.zero_grad()
            predict = self.model[list(self.model.keys())[0]](inp)
            step_one_loss = self.loss(predict, tar_step1)
            step_one_loss.backward()
            optimizer.step()
            self.last_step_timing = None
            return {self.loss_type: step_one_loss.item()}

        timing = {}
        phase_names = ("preprocess_h2d", "zero_grad", "forward", "loss", "backward", "optimizer")
        cuda_events = []

        def record_event():
            return _ACCELERATOR.record_event()

        compute_start = time.perf_counter()
        cuda_events.append(record_event())

        phase_start = time.perf_counter()
        inp, tar_step1 = self.data_preprocess(batch_data)
        timing["preprocess_h2d_host_enqueue"] = time.perf_counter() - phase_start
        cuda_events.append(record_event())

        optimizer = self.optimizer[list(self.model.keys())[0]]
        phase_start = time.perf_counter()
        optimizer.zero_grad()
        timing["zero_grad_host_enqueue"] = time.perf_counter() - phase_start
        cuda_events.append(record_event())

        phase_start = time.perf_counter()
        predict = self.model[list(self.model.keys())[0]](inp)
        timing["forward_host_enqueue"] = time.perf_counter() - phase_start
        cuda_events.append(record_event())

        phase_start = time.perf_counter()
        step_one_loss = self.loss(predict, tar_step1)
        timing["loss_host_enqueue"] = time.perf_counter() - phase_start
        cuda_events.append(record_event())

        phase_start = time.perf_counter()
        step_one_loss.backward()
        timing["backward_host_enqueue"] = time.perf_counter() - phase_start
        cuda_events.append(record_event())

        phase_start = time.perf_counter()
        optimizer.step()
        timing["optimizer_host_enqueue"] = time.perf_counter() - phase_start
        cuda_events.append(record_event())

        # This synchronization already existed in the original training path.
        # It completes the queued default-stream work without adding a new sync.
        phase_start = time.perf_counter()
        loss_value = step_one_loss.item()

        timing["loss_item_natural_sync"] = time.perf_counter() - phase_start
        timing["compute_host_total"] = time.perf_counter() - compute_start
        events_ready = _ACCELERATOR.event_ready(cuda_events[-1])
        for index, name in enumerate(phase_names):
            value = float("nan")
            if events_ready:
                elapsed = _ACCELERATOR.event_elapsed_seconds(
                    cuda_events[index], cuda_events[index + 1])
                if elapsed is not None:
                    value = elapsed
            # Keep the historical key name for existing H100 log parsers.
            timing[name + "_cuda"] = value
        timing["cuda_events_ready"] = events_ready
        timing["mode"] = "accelerator_events_no_extra_sync"
        self.last_step_timing = timing

        return {self.loss_type: loss_value}



    def test_one_step(self, batch_data):
        inp, tar_step1,  = self.data_preprocess(batch_data)

        predict = self.model[list(self.model.keys())[0]](inp)

        step_one_loss = self.loss(predict, tar_step1)

        loss = step_one_loss
        return {self.loss_type: loss.item()}

