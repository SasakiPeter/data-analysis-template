import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger import setup_logger  # NOQA


__all__ = ('hoge',)


def hoge():
    logger = setup_logger(__name__)
    logger.info('hoge')


def fuga():
    logger = setup_logger(__name__)
    logger.info('fuga')


if __name__ == '__main__':
    hoge()
    fuga()
