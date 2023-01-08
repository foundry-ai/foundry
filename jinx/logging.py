from loguru import logger as uru_logger

# A logger which can also be used
# inside of jax functions using id_tap
# callbacks. This wraps the loguru logger
class JaxLogger:
    def __init__(self, logger):
        self.logger = logger
    
    def info(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

logger = JaxLogger(uru_logger)