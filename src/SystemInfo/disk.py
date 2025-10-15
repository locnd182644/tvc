import psutil

def get_disk_info():
    try:
        disk = psutil.disk_usage('/')
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        }
    except Exception as e:
        return {
            "error": str(e)
        }