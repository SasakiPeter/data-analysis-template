from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO

# loggers = {}


def setup_logger(name, logfile='debug.log'):
    # global loggers

    # if loggers.get(name):
    #     return loggers.get(name)

    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.propagate = False

    if len(logger.handlers) == 2:
        return logger

    # INFO Handler
    info_handler = StreamHandler()
    info_fmt = Formatter(
        '%(asctime)s %(name)s[%(funcName)s]: [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(info_fmt)
    info_handler.setLevel(INFO)

    # DEBUG Handler as file
    debug_handler = FileHandler(logfile, 'a')
    debug_fmt = Formatter(
        '%(asctime)s %(filename)s %(name)s[%(funcName)s]: [%(levelname)s] %(message)s')
    debug_handler.setFormatter(debug_fmt)
    debug_handler.setLevel(DEBUG)

    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)
    # loggers[name] = logger
    return logger


if __name__ == '__main__':
    class Horse:
        def __init__(self):
            self.logger = setup_logger(self.__class__.__name__)

        def scream(self):
            self.logger.info('ヒヒーン！')
    logger = setup_logger(__name__)
    logger.info("INFO message")
    logger.debug("debug message")
    h1 = Horse()
    h2 = Horse()
    h2.scream()
