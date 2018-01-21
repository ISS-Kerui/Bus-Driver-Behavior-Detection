# -*- coding: utf-8 -*-

import logging
import time
import os
"""
author: ISS-Kerui
date:2018-01-20
Define log format
"""
class Logger():
    def __init__(self, logsite=''):
        # get the root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        while 0 != len(self.logger.handlers):
            self.logger.removeHandler(self.logger.handlers[0])

        # create a handler of the log file
        logpath = logsite + 'Logs'
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        filename = time.strftime('%Y%m%d',time.localtime(time.time()))
        filename = logpath + '/log_' + filename + '.log'
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)

        # create a handler of the console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # define the output format
        formatter = logging.Formatter('[%(asctime)s - %(filename)s - %(levelname)s] <%(message)s> [%(funcName)s - %(lineno)d][pID:%(process)d - tID:%(thread)d]')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger
