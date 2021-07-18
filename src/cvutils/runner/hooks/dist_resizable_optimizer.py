from src.cvcls.core.utils.dist_utils import DistOptimizerHook

from .hook import HOOKS


@HOOKS.register_module()
class DistResizableOptimizerHook(DistOptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.step()
    
    def within_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()