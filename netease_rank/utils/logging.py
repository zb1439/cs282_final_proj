import logging
import os
import sys


logging.basicConfig(filename=os.path.join(os.getcwd(), 'log.txt'), level="INFO")
logger = logging.getLogger("NetEase")
h = logging.StreamHandler(sys.stdout)
h.flush = sys.stdout.flush
logger.addHandler(h)
