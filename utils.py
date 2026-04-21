import os
import datetime
import socket
import logging
import petname


def initialize_logging(app_name, logdir="logs", debug=False, run_id=None):
    debugtag = "-debug" if debug else ""
    logtag = petname.Generate(2)
    username = os.path.split(os.path.expanduser("~"))[-1]
    hostname = socket.gethostname().replace(".stanford.edu", "")
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    starttimestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO if not debug else logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{logdir}/{app_name}{debugtag}_{logtag}_{username}_{hostname}_{starttimestr}.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized for '{app_name}' by '{username}' on host '{hostname}' with ID '{logtag}'")
    return username, hostname, logtag, starttimestr