import psutil

def get_memory_info():
    """
    Retrieve memory information of the system.

    Returns:
        dict: A dictionary containing total, available, used memory and the percentage of memory used.
    """
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent,
    }