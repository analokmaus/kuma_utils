class DummyGradScaler:
    '''
    A dummy scaler with the same interface as amp.Gradscaler
    '''
    def scale(self, loss):
        return loss
    
    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


class DummyAutoCast:
    '''
    '''
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
