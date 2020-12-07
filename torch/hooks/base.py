
class HookTemplate:
    '''
    Hook is called in each mini-batch during traing / inference 
    and after processed all mini-batches, 
    in order to define the training process and evaluate the results of each epoch.
    '''

    def __init__(self):
        pass

    def forward_train(self, trainer, inputs):
        # return approx, target, loss, metric, extra
        pass

    def forward_test(self, trainer, inputs):
        # return approx
        pass

    def evaluate_epoch(self, trainer, approxs, targets, extras):
        # return metric_total, monitor_metrics_total
        pass
