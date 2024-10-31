import torch
from .base import HookTemplate
from ..clip_grad import dispatch_clip_grad


class TrainHook(HookTemplate):

    def __init__(self, evaluate_in_batch=False, clip_grad=None, max_grad_norm=100., sam_optimizer=False):
        super().__init__()
        self.evaluate_in_batch = evaluate_in_batch
        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm
        self.sam_optimizer = sam_optimizer

    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        approx = trainer.model(*inputs[:-1])
        loss = trainer.criterion(approx, target)
        return loss, approx.detach()

    forward_valid = forward_train

    def forward_test(self, trainer, inputs):
        approx = trainer.model(*inputs[:-1])
        return approx
    
    def _backprop_normal(self, trainer, loss, inputs=None):
        trainer.scaler.scale(loss).backward()
        dispatch_clip_grad(trainer.model.parameters(), self.max_grad_norm, mode=self.clip_grad)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad()

    def _backprop_sam(self, trainer, loss, inputs):
        trainer.scaler.scale(loss).backward()
        dispatch_clip_grad(trainer.model.parameters(), self.max_grad_norm, mode=self.clip_grad)
        if trainer.fp16:
            # first step
            optimizer_state = trainer.scaler._per_optimizer_states[id(trainer.optimizer)]
            trainer.scaler.unscale_(trainer.optimizer)
            if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                trainer.optimizer.first_step(zero_grad=True)
            optimizer_state["stage"] = 2
            trainer.scaler.update()
            # second step
            with trainer.autocast:
                loss2, _ = trainer.forward_train(trainer, inputs)
            trainer.scaler.scale(loss2).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                trainer.optimizer.second_step(zero_grad=True)
            optimizer_state["stage"] = 2
        else:
            trainer.optimizer.first_step(zero_grad=True)
            loss2, _ = trainer.forward_train(trainer, inputs)
            loss2.backward()
            trainer.optimizer.second_step(zero_grad=True)
        trainer.scaler.update()
        trainer.optimizer.zero_grad()

    def backprop(self, trainer, loss, inputs=None):
        if self.sam_optimizer:
            self._backprop_sam(trainer, loss, inputs)
        else:
            self._backprop_normal(trainer, loss, inputs)

    def _evaluate(self, trainer, approx, target):
        if trainer.eval_metric is None:
            if trainer.criterion is None:
                metric_score = 0.
            else:
                metric_score = trainer.criterion(approx, target).item()
        else:
            metric_score = trainer.eval_metric(approx, target)
            if isinstance(metric_score, torch.Tensor):
                metric_score = metric_score.item()
        monitor_score = []
        for monitor_metric in trainer.monitor_metrics:
            score = monitor_metric(approx, target)
            if isinstance(score, torch.Tensor):
                score = score.item()
            monitor_score.append(score)
        return metric_score, monitor_score

    def evaluate_batch(self, trainer, inputs, approx):
        target = inputs[-1]
        storage = trainer.epoch_storage
        if self.evaluate_in_batch:
            # Add scores to storage
            metric_score, monitor_score = self._evaluate(trainer, approx, target)
            storage['batch_metric'].append(metric_score)
            storage['batch_monitor'].append(monitor_score)
        else:
            # Add prediction and target to storage
            storage['approx'].append(approx)
            storage['target'].append(target)

    def evaluate_epoch(self, trainer):
        storage = trainer.epoch_storage
        if self.evaluate_in_batch:
            # Calculate mean metrics from all batches
            metric_total = storage['batch_metric'].mean(0)
            monitor_total = storage['batch_monitor'].mean(0).tolist()
        else:
            # Calculate scores
            metric_total, monitor_total = self._evaluate(
                trainer, storage['approx'], storage['target'])
        return metric_total, monitor_total


SimpleHook = TrainHook  # Compatibility
