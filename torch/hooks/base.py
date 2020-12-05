
class HookTemplate:
    '''
    Hook is called in each mini-batch during traing / inference 
    and after processed all mini-batches, 
    in order to define the training process and evaluate the results of each epoch.
    '''

    def __init__(self):
        pass

    def batch_train(self, trainer, inputs):
        pass

    def batch_test(self, trainer, inputs):
        pass

    def epoch_eval(self, trainer, approxs, targets, extras):
        pass
