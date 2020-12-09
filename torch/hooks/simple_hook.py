from .base import HookTemplate


class SimpleHook(HookTemplate):

    def __init__(self, evaluate_batch=False):
        super().__init__()
        self.evaluate_batch = evaluate_batch

    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        approx = trainer.model(inputs[0])
        loss = trainer.criterion(approx, target)
        
        storage = trainer.epoch_storage
        if self.evaluate_batch:
            metric = trainer.eval_metric(approx, target)
            monitor_metrics_total = []
            for monitor_metric in trainer.monitor_metrics:
                monitor_metrics_total.append(
                    monitor_metric(approx, target))
            storage['batch_metric'].append(metric)
            storage['batch_monitor'].append(monitor_metrics_total)
        storage['approx'].append(approx)
        storage['target'].append(target)
        return approx, target, loss

    def forward_test(self, trainer, inputs):
        approx = trainer.model(inputs[0])
        return approx

    def evaluate_epoch(self, trainer):
        storage = trainer.epoch_storage
        if self.evaluate_batch: # Batch level evaluation
            metric_total = storage['batch_metric'].mean(0)
            monitor_metrics_total = storage['batch_monitor'].mean(0).tolist()

        else: # Dataset level evaluation
            if trainer.eval_metric is None:
                metric_total = None
            else:
                metric_total = trainer.eval_metric(
                    storage['approx'], storage['target'])
                
            monitor_metrics_total = []
            for monitor_metric in trainer.monitor_metrics:
                monitor_metrics_total.append(
                    monitor_metric(storage['approx'], storage['target']))
        return metric_total, monitor_metrics_total
