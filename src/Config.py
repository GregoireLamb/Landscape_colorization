import yaml
import torch
class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
            self.use_gpu = self.config.get('use_gpu', False) and torch.cuda.is_available()

    def __getattr__(self, name):
        return self.config.get(name, None)
