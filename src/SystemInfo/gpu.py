import GPUtil

def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        return {"error": str(e)}
    gpu_list = [{
        "name": gpu.name,
        "memory_total": gpu.memoryTotal,
        "memory_used": gpu.memoryUsed,
        "memory_free": gpu.memoryFree,
        "load": gpu.load * 100,
        "temperature": gpu.temperature,
    } for gpu in gpus]
    return gpu_list