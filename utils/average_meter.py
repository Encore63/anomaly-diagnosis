class AverageMeter(object):
    def __init__(self, info='AverageMeter'):
        self.cnt = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = info
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
