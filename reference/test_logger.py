import logging, datetime, sys

format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
formatter = logging.Formatter(format)


def setup_logging(folder, log_std_out=True):
    global LOGGER
    # Set up logging

    format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    format = {"fmt":'%(asctime)s %(levelname)s %(message)s', "datefmt":"%H:%M:%S"}

    # Override Pycharm logging - otherwise, logging file may not be created
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger("root." + __name__)

    today = datetime.datetime.now()
    log_path = "{}/{}.log".format(folder, today.strftime("%m-%d-%Y"))
    if folder is None:
        log_path = None
    logging.basicConfig(filename=log_path,
                            filemode='a',
                            format=format["fmt"],
                            datefmt=format["datefmt"],
                            level=logging.INFO)

    # Send log messages to standard out
    if log_std_out:
        formatter = logging.Formatter(**format)
        std_out = logging.StreamHandler(sys.stdout)
        std_out.setLevel("INFO")
        std_out.setFormatter(formatter)
        logger.addHandler(std_out)

    LOGGER = logger
    return logger

if __name__=="__main__":
    logger = setup_logging(".")
    print("SUCCESS")
    logger.info("TEST")

