import loguru


class _Logger:
    def __init__(self):
        self._logger = loguru.logger

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return self.loguru.logger(attr, *args, **kwargs)
        return wrapper
