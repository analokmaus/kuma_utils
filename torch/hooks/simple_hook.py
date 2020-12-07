from .base import HookTemplate


class SimpleHook(HookTemplate):

    def __init__(self):
        super().__init__()

    def forward_train(self, trainer, inputs):
        ''' Forward '''
        target = inputs[-1]
        approx = trainer.model(inputs[0])
        loss = trainer.criterion(approx, target)
        metric = None
        extra = None
        return approx, target, loss, metric, extra

    def forward_test(self, trainer, inputs):
        approx = trainer.model(inputs[0])
        return approx

    def evaluate_epoch(self, trainer, approxs, targets, extras):
        if trainer.eval_metric is None:
            metric_total = None
        elif len(extras) > 0:
            metric_total = trainer.eval_metric(approxs, targets, extras)
        else:
            metric_total = trainer.eval_metric(approxs, targets)
            
        monitor_metrics_total = []
        for monitor_metric in trainer.monitor_metrics:
            if len(extras) > 0:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets, extras))
            else:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets))
        return metric_total, monitor_metrics_total
