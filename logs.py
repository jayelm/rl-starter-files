"""
Logger class
"""

import logging
import tqdm


logging.basicConfig(
    format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(name):
    log = logging.getLogger(name)
    #  log.addHandler(TqdmLoggingHandler())
    return log
