#!/usr/bin/env python
# coding: utf-8

import signal
import sys
from glob import glob
from pathlib import Path

import ray
from loguru import logger


def keyboard_interrupt_handler(sig: int, _) -> None:
    """This function handles the KeyboardInterrupt (CTRL+C) signal.
    It's a handler for the signal, which means it's called when the OS sends
    the signal. The signal is sent when the user presses CTRL+C.

    Parameters
    ----------
    The function takes two arguments:
    sig:
        The id of the signal that was sent.
    frame:
        The current stack frame.
    """
    logger.warning(f'KeyboardInterrupt (id: {sig}) has been caught...')
    logger.info('Terminating the session gracefully...')
    ray.shutdown()
    minio_leftovers = glob('*.part.minio')
    for leftover in minio_leftovers:
        Path(leftover).unlink()
    sys.exit(1)


def catch_keyboard_interrupt():
    """This function catches the keyboard interrupt handler."""
    return signal.signal(signal.SIGINT, keyboard_interrupt_handler)
