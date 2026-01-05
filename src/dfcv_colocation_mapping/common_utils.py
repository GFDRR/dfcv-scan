import yaml


def read_config(config_file: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data as a dictionary.
    """
    try:
        # Open the YAML configuration file in read mode
        with open(config_file, "r") as file:
            # Parse the YAML content into a Python dictionary
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        # Raise error if the YAML is invalid
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config
