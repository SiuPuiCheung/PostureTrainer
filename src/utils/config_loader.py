"""Configuration loader for posture trainer."""
# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration management for posture trainer."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file. 
                        Defaults to config/config.yaml in project root.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config['model']
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get path configuration."""
        return self._config['paths']
    
    @property
    def analysis_types(self) -> list:
        """Get analysis type configurations."""
        return self._config['analysis']['types']
    
    @property
    def body_labels(self) -> Dict[str, list]:
        """Get body labels for different analysis types."""
        return self._config['body_labels']
    
    @property
    def gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration."""
        return self._config['gui']
    
    @property
    def report_config(self) -> Dict[str, Any]:
        """Get report configuration."""
        return self._config['report']

    @property
    def pose_models(self) -> Dict[str, Any]:
        """Get pose model configuration."""
        return self._config.get('pose_models', {})

    def get_pose_model_options(self) -> list:
        """Return configured pose model options list."""
        pose_cfg = self.pose_models
        if isinstance(pose_cfg, dict):
            options = pose_cfg.get('options', [])
            if isinstance(options, list):
                return options
        return []

    def get_default_pose_model(self) -> str:
        """Return default pose model identifier."""
        pose_cfg = self.pose_models
        if isinstance(pose_cfg, dict):
            default_model = pose_cfg.get('default')
            if isinstance(default_model, str):
                return default_model
        options = self.get_pose_model_options()
        if options:
            opt = options[0]
            if isinstance(opt, dict) and 'id' in opt:
                return opt['id']
        return 'mediapipe'

    def get_default_pose_device(self) -> str:
        """Return default compute device identifier (auto/cpu/gpu)."""
        pose_cfg = self.pose_models
        if isinstance(pose_cfg, dict):
            default_device = pose_cfg.get('default_device')
            if isinstance(default_device, str):
                return default_device
        return 'auto'

    def get_pose_weights_dir(self) -> str:
        """Return directory path for storing pose model weights."""
        pose_cfg = self.pose_models
        if isinstance(pose_cfg, dict):
            weights_dir = pose_cfg.get('weights_dir')
            if isinstance(weights_dir, str) and weights_dir:
                return weights_dir
        return 'models'
    
    def get_body_labels_by_index(self, index: int) -> list:
        """
        Get body labels for a specific analysis type by index.
        
        Args:
            index: Index of the analysis type (0-5).
        
        Returns:
            List of body labels for the specified analysis type.
        """
        label_keys = ['front_angle', 'side_angle', 'balance_back', 
                     'balance_test', 'balance_front', 'balance_side']
        if 0 <= index < len(label_keys):
            return self.body_labels[label_keys[index]]
        return []
