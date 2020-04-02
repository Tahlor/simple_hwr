import logging, datetime, sys
from pathlib import Path

""" This is a custom logging class
    Will log to file and print out if desired
    
    Usage:
        from hwr_logger import logger # now you have a logger!
    
    Setup logger:
        hwr_logger.setup_logging(*args) # the logger has been updated everywhere with new folder etc. since Python only imports 1x!
        
    Create sub logger:
        import logging
        logger = logging.getLogger("root."+__name__) # this will inherit from our root logger!
            
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")

d = {"DEBUG":logging.DEBUG,
     "INFO":logging.INFO,
     "WARNING":logging.WARNING,
     "ERROR": logging.ERROR,
     "CRITICAL": logging.CRITICAL,
     }

def setup_logging(folder=None, log_std_out=True, level=logging.INFO):
    global logger
    # Set up logging
    if isinstance(level, str):
        level = d[level.upper()]

    if folder is None:
        folder = Path("./logs")
    else:
        folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    format = {"fmt": '%(asctime)s %(levelname)s %(message)s', "datefmt": "%H:%M:%S"}

    # Override Pycharm logging - otherwise, logging file may not be created
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #logger = logging.getLogger(__name__)

    today = datetime.datetime.now()
    log_path = "{}/{}.log".format(folder.as_posix(), today.strftime("%m-%d-%Y"))

    logging.basicConfig(filename=log_path,
                        filemode='a',
                        format=format["fmt"],
                        datefmt=format["datefmt"],
                        level=level)

    # Send log messages to standard out
    if log_std_out:
        formatter = logging.Formatter(**format)
        std_out = logging.StreamHandler(sys.stdout)
        std_out.setLevel(level)
        std_out.setFormatter(formatter)
        logger.addHandler(std_out)

    return logger