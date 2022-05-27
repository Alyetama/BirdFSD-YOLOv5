#!/usr/bin/env python
# coding: utf-8

import signal
import sys
from glob import glob
from pathlib import Path
from typing import Callable

import ray
from loguru import logger


def keyboard_interrupt_handler(sig: int, _: object) -> None:
    """This function handles the KeyboardInterrupt (CTRL+C) signal.

    It's a handler for the signal, which means it's called when the OS sends
    the signal. The signal is sent when the user presses CTRL+C.

    Args:
        sig (int): The id of the signal that was sent.
        _ (object): The current stack frame.

    Returns:
        None

    """
    logger.warning(f'KeyboardInterrupt (id: {sig}) has been caught...')
    logger.info('Terminating the session gracefully...')
    ray.shutdown()
    minio_leftovers = glob('*.part.minio')
    for leftover in minio_leftovers:
        Path(leftover).unlink()
    try:
        Path('best.pt').unlink()
    except FileNotFoundError:
        pass
    sys.exit(1)


def catch_keyboard_interrupt() -> Callable:
    """This function catches the keyboard interrupt handler.

    Returns:
        Callable: A keyboard interrupt handler callable

    """
    return signal.signal(signal.SIGINT, keyboard_interrupt_handler)
