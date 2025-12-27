from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FreezeModulesHook(Hook):
    """Freeze specific modules during training.

    Args:
        modules (list[str]): List of module names to freeze.
    """

    def __init__(self, modules):
        self.modules = modules

    def before_train(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        for module_name in self.modules:
            module = dict(model.named_modules())[module_name]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        # 打印冻结信息
        print(f"Froze modules: {self.modules}")


@HOOKS.register_module()
class UpdateJointAngleClassificationHeadTemp(Hook):
    """Update the Temperature of Joint Angle Classification Head.

    Args:
        modules (list[str]): List of module names to freeze.
    """
    def __init__(self, module_name):
        self.module_name = module_name
        print(f"module name: {module_name}")

    def after_train_iter(self, runner, batch_idx: int, 
                         data_batch = None, 
                         outputs = None) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        target_module = dict(model.named_modules())[self.module_name]
        if getattr(target_module, 'temperature', None) is not None and hasattr(target_module.temperature, 'update'):
            target_module.temperature.update()
            if batch_idx == 0:
                print(f"Update temperature of {self.module_name} to {target_module.temperature.get_temperature()}")
