import gin.tf


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.
    Args:
      gin_files: list, of paths to the gin configuration files for this
        experiment.
      gin_bindings: list, of gin parameter bindings to override the values in
        the config files.
    """
    gin.parse_config_files_and_bindings(
        gin_files, bindings=gin_bindings, skip_unknown=False
    )
