import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict

class ConfigManager:
    def __init__(self):
        self.config = {}
        self.secrets = {}
        self._load_config()
        self._load_secrets()
        
    def _load_config(self):
        config_path = Path(__file__).parent.parent.parent / 'config' / 'settings.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
    def _load_secrets(self):
        load_dotenv()
        secrets_path = Path(__file__).parent.parent.parent / 'config' / 'secrets.yaml'
        with open(secrets_path) as f:
            raw_secrets = yaml.safe_load(f)
            self.secrets = self._resolve_env_vars(raw_secrets)
    
    def _resolve_env_vars(self, config: Dict) -> Dict:
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]
                        config[section][key] = os.getenv(env_var)
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        try:
            keys = key_path.split('/')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return self._get_secret(key_path, default)
    
    def _get_secret(self, key_path: str, default: Any) -> Any:
        try:
            keys = key_path.split('/')
            value = self.secrets
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

# Singleton instance
config_manager = ConfigManager()