#!/usr/bin/env python
# coding: utf-8

from loguru import logger as _logger


class logger:

    def add(x):
        _logger.add(x)

    def debug(x):
        _logger.debug(f'\33[34m{x}\x1b[0m')

    def info(x):
        _logger.info(f'\33[32m{x}\x1b[0m')

    def warning(x):
        _logger.warning(f'\33[33m{x}\x1b[0m')

    def error(x):
        _logger.error(f'\33[101m{x}\x1b[0m')

    def exception(x):
        _logger.exception(x)
