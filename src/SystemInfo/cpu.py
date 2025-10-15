import psutil

def get_cpu_info():
    """
    Retrieve CPU information including physical cores, total cores, frequency, and usage.

    Returns:
        dict: A dictionary containing CPU information.
    """
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "cpu_usage": psutil.cpu_percent(interval=1),
    }