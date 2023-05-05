import inspect
import yaml

from typing import Dict, Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class Config_Trainer:
    __doc__ = r"""
            Configure the arguments for pl.Trainer, from the given .yaml file or its DictConfig contents.

            Examples:    
                Trainer:
                      accelerator: gpu
                      devices: 1
                      max_epochs: 10
                      gradient_clip_val: 1.0

                Callbacks:
                    EarlyStopping:
                        monitor: val_acc
                        patience: 3
                Logger:
                    WandbLogger:
                        name: exp_0
                        project: KoBigBird-nsmc

                cfg_trainer = Config_Trainer(default.yaml)()
                trainer = pl.Trainer(**cfg_trainer)

            * A category must be equal to 'config_set'
            * The class must be given as it is in the official docs, excluding the parent path;
            https://pytorch-lightning.readthedocs.io/en/stable/api_references.html
        """

    config_set = {'Trainer', 'Callbacks', 'Profiler', 'Logger', 'Strategy', 'Plugins'}

    def __init__(self, cfg: Union[DictConfig, str]):
        if isinstance(cfg, DictConfig):
            self.cfg = cfg

        elif isinstance(cfg, str):
            with open(cfg) as f:
                self.cfg = yaml.load(f, yaml.FullLoader)
        else:
            raise TypeError('config should be given in dictionary or yaml file')

        if self.cfg:
            if not set(self.cfg.keys()).issubset(self.config_set):
                raise MisconfigurationException(f"please configure among {self.config_set}")

            if 'Profiler' in self.cfg and 'profiler' in self.cfg['Trainer']:
                raise MisconfigurationException("'profiler' is already configured in Trainer argument")

            if 'Logger' in self.cfg and 'logger' in self.cfg['Trainer']:
                raise MisconfigurationException("'logger' is already configured in Trainer argument")

            if 'Strategy' in self.cfg and 'strategy' in self.cfg['Trainer']:
                raise MisconfigurationException("'strategy' is already configured in Trainer argument")

            self.callbacks = self.config_callbacks()
            self.profiler = self.config_profiler()
            self.logger = self.config_loggers()
            self.strategy = self.config_strategy()
            self.plugins = self.config_plugins()

        else:
            self.cfg = {'Trainer': {}}
            self.callbacks, self.profiler, self.logger, self.strategy, self.plugins \
                = None, None, None, None, None


    def __call__(self) -> Dict:
        kwargs = {**self.cfg['Trainer']}

        if self.callbacks:
            kwargs['callbacks'] = self.callbacks
        if self.profiler:
            kwargs['profiler'] = self.profiler
        if self.logger:
            kwargs['logger'] = self.logger
        if self.strategy:
            kwargs['strategy'] = self.strategy
        if self.plugins:
            kwargs['plugins'] = self.plugins

        return kwargs


    def config_module(self, cfg_module, target):
        module, params = list(self.cfg[cfg_module].items())[0]
        module_dict = {n: m for n, m in inspect.getmembers(target)}

        if module in module_dict:
            if not params:
                params = {}
            module = module_dict[module](**params)
        else:
            raise ValueError(f'No {module} in {target}')

        return module


    def config_modules(self, cfg_module, target):
        module_dict = {n: m for n, m in inspect.getmembers(target)}
        modules = []

        for module, params in self.cfg[cfg_module].items():
            if module in module_dict:
                if not params:
                    params = {}
                modules.append(module_dict[module](**params))
            else:
                raise ValueError(f'No {module} in {target}')

        return modules


    def config_profiler(self):
        if 'Profiler' in self.cfg:
            import pytorch_lightning.profiler as Profiler
            return self.config_module('Profiler', Profiler)
        else:
            return None

    def config_strategy(self):
        if 'Strategy' in self.cfg:
            import pytorch_lightning.strategies as Strategy
            return self.config_module('Strategy', Strategy)
        else:
            return None

    def config_callbacks(self):
        if 'Callbacks' in self.cfg:
            import pytorch_lightning.callbacks as Callbacks
            return self.config_modules('Callbacks', Callbacks)
        else:
            return None

    def config_loggers(self):
        if 'Logger' in self.cfg:
            import pytorch_lightning.loggers as Logger
            return self.config_modules('Logger', Logger)
        else:
            return None

    def config_plugins(self):
        if 'Plugins' in self.cfg:
            import pytorch_lightning.plugins as Plugins
            return self.config_modules('Plugins', Plugins)
        else:
            return None