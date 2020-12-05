from .base import HookTemplate


class ArgumentSpecifiedHook(HookTemplate):

    def __init__(self, 
                 argument_target=-1, argument_to_model=[0], 
                 argument_to_criterion=None, argument_extra=None):
        super().__init__()
        self.argument_target = argument_target
        self.argument_to_model = argument_to_model
        if not isinstance(self.argument_to_model, (list, tuple)):
            self.argument_to_model = [self.argument_to_model]
        self.argument_to_criterion = argument_to_criterion
        self.argument_extra = argument_extra

    def batch_train(self, model, inputs, criterion, eval_metric):
        ''' Forward '''
        target = inputs[self.argument_target]
        approx = model(*[inputs[i] for i in self.argument_to_model])
        if self.argument_to_criterion is not None:
            loass = criterion(approx, target, inputs[self.argument_to_criterion])
        else:
            loss = criterion(approx, target)
        metric = None
        if self.argument_extra is not None:
            extra = inputs[self.argument_extra]
        else:
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
        approx = model(*[inputs[i] for i in self.argument_to_model])
        return approx
