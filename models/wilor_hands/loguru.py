import logging


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class _Logger:
    def __init__(self):
        self._logger = logging.getLogger("wilor_hmp")

    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)


logger = _Logger()
