import logging

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logging.basicConfig(fotmat=FORMAT)
logger = logging.getLogger(name='Tensorflow-Vegetable')
