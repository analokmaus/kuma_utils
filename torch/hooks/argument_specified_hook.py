from .base import HookTemplate


class ArgumentSpecifiedHook(HookTemplate):

    def __init__(self, evaluate_batch=False, 
                 argument_target=-1, argument_to_model=[0], 
                 argument_to_criterion=None, argument_extra=None):
        super().__init__()
        self.evaluate_batch = evaluate_batch
        self.argument_target = argument_target
        self.argument_to_model = argument_to_model
        if not isinstance(self.argument_to_model, (list, tuple)):
            self.argument_to_model = [self.argument_to_model]
        self.argument_to_criterion = argument_to_criterion
        self.argument_extra = argument_extra

    def forward_train(self, trainer, inputs):
        target = inputs[self.argument_target]
        approx = trainer.model(*[inputs[i] for i in self.argument_to_model])
        
        if self.argument_to_criterion is not None:
            loss = trainer.criterion(
                approx, target, inputs[self.argument_to_criterion])
        else:
            loss = trainer.criterion(approx, target)

        if self.evaluate_batch:
            if self.argument_extra is not None:
                metric = trainer.eval_metric(approx, target, inputs[self.argument_extra])
            else: 
                metric = trainer.eval_metric(approx, target)
            monitor_metrics_total = []
            for monitor_metric in trainer.monitor_metrics:
                if self.argument_extra is not None:
                    monitor_metrics_total.append(
                        monitor_metric(approx, target, inputs[self.argument_extra]))
                else:
                    monitor_metrics_total.append(
                        monitor_metric(approx, target))
            trainer.epoch_storage['batch_metric'].append(metric)
            trainer.epoch_storage['batch_monitor'].append(monitor_metrics_total)
        trainer.epoch_storage['approx'].append(approx)
        trainer.epoch_storage['target'].append(target)
        if self.argument_extra is not None:
            trainer.epoch_storage['extra'].append(inputs[self.argument_extra])
        return approx, target, loss

    def forward_test(self, trainer, inputs):
        approx = trainer.model(*[inputs[i] for i in self.argument_to_model])
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
                if len(storage['extra']) > 0:
                    metric_total = trainer.eval_metric(
                        storage['approx'], storage['target'], storage['extra'])
                else:
                    metric_total = trainer.eval_metric(
                        storage['approx'], storage['target'])
                
                monitor_metrics_total = []
                for monitor_metric in trainer.monitor_metrics:
                    if len(storage['extra']) > 0:
                        monitor_metrics_total.append(
                            monitor_metric(storage['approx'], storage['target'], 
                                           storage['extra']))
                    else:
                        monitor_metrics_total.append(
                            monitor_metric(storage['approx'], storage['target']))
        return metric_total, monitor_metrics_total
