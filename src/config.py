"""Configuration loading.

Every experiment is described by a small YAML file.  ``configs/base.yaml`` holds
the settings that are shared by the two released tasks; a task file overrides
the entries it needs.  All paths are resolved relative to the repository root so
that the code can be moved or cloned anywhere.
"""

import copy
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class Config(dict):
    """Nested configuration with dotted access to the leaves."""

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted):
        sentinel = object()
        value = self.get_path(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing configuration entry: {dotted}")
        return value

    @property
    def data_root(self):
        return self._absolute(self.require('data.root'))

    @property
    def output_root(self):
        return self._absolute(self.require('data.output_root'))

    def _absolute(self, path):
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))

    def dataset_dir(self, dataset):
        """Folder of ``dataset``, tolerating a different capitalisation."""
        path = os.path.join(self.data_root, dataset)
        if os.path.isdir(path):
            return path
        if os.path.isdir(self.data_root):
            for entry in sorted(os.listdir(self.data_root)):
                if entry.lower() == dataset.lower() and os.path.isdir(os.path.join(self.data_root, entry)):
                    return os.path.join(self.data_root, entry)
        return path

    def output_dir(self, *parts):
        return os.path.join(self.output_root, *parts)


def load_config(task_config, base_config=None):
    """Load a task configuration, optionally merged on top of a base one."""
    if base_config is None:
        base_config = os.path.join(os.path.dirname(task_config), 'base.yaml')

    config = {}
    if os.path.exists(base_config):
        with open(base_config, 'r') as handle:
            config = yaml.safe_load(handle) or {}

    with open(task_config, 'r') as handle:
        task = yaml.safe_load(handle) or {}

    return Config(_deep_update(copy.deepcopy(config), task))
