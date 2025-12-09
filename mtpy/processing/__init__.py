from loguru import logger

try:
    from .aurora.process_aurora import AuroraProcessing
except ImportError:
    msg = "Import from mtpy.processing.aurora failed"
    msg = f"{msg} This is a known issue when aurora imports from mtpy"
    logger.debug(msg)

__all__ = ["AuroraProcessing"]
