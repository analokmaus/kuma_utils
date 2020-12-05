
class HookTemplate:
    '''
    Hook is called in each batch training / inference, 
    in order to define the training process and carry out
    epoch level evaluation.
    '''

    def __init__(self):
        pass

    def batch_train(self, model, inputs, criterion, eval_metric, logger, tb_logger):
        pass

    batch_eval = batch_train

    def end_train_eval(self, approxs, targets, extras, eval_metric, monitor_metrics):
        pass

    def batch_test(self, model, inputs):
        pass
