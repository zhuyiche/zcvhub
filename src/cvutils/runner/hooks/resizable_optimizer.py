
from torch.nn.utils import clip_grad

from .hook import HOOKS, Hook
from .optimizer import OptimizerHook


@HOOKS.register_module()
class ResizableOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.step()
    
    def within_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()