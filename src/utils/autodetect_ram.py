import psutil


def auto_select_model():
    """Selecting model depending on available ram"""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb >= 12:
        return "qwen3:8b"
    elif ram_gb >= 6:
        return "mistral:7b"
    else:
        return "phi4-mini"
