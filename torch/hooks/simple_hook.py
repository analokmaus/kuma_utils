from .base import HookTemplate


class SimpleTrainEval(HookTemplate):

    def __init__(self):
        super().__init__()

    def batch_train(self, model, inputs, criterion, eval_metric):
        ''' Forward '''
        target = inputs[-1]
        approx = model(inputs[0])
        loss = criterion(approx, target)
        metric = None
        extra = None
        return approx, target, loss, metric, extra

    def end_train_eval(self, approxs, targets, extras, 
                       eval_metric, monitor_metrics):
        if eval_metric is None:
            metric_total = None
        elif len(extras) > 0:
            metric_total = eval_metric(approxs, targets, extras)
        else:
            metric_total = eval_metric(approxs, targets)
            
        monitor_metrics_total = []
        for monitor_metric in monitor_metrics:
            if len(extras) > 0:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets, extras))
            else:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets))
        return metric_total, monitor_metrics_total

    def batch_test(self, model, inputs):
        approx = model(inputs[0])
        return approx
