import yaml

class Config:
    def __init__(self):
        self.config = {}

    def load(self, config_path):
        with open(config_path, 'r') as file:
            new_config = yaml.safe_load(file)
            if new_config: 
                self.config.update(new_config)

    def get(self, key, default=None):
        return self.config.get(key, default)

config = Config()

