from submodules import hoge
from logger import setup_logger


def main():
    logger = setup_logger(__name__)
    logger.info(hoge())


if __name__ == '__main__':
    main()
