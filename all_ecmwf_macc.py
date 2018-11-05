#!/usr/bin/env python

from datetime import datetime
import logging
import os
import threading

from ecmwfapi import ECMWFDataServer

logging.basicConfig()
logger = logging.getLogger('ecmwf')
logger.setLevel(logging.DEBUG)


def ecmwf(param, targetname, startdate, stopdate):
    server.retrieve({
        "class": "mc",
        "dataset": "macc",
        "date": "%s/to/%s" % (startdate, stopdate),
        "expver": "rean",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": param,
        "step": "3/6/9/12/15/18/21/24",
        "stream": "oper",
        "format": "netcdf",
        "time": "00:00:00",
        "type": "fc",
        "target": targetname,
    })    


params = {
    "aod1240": "216.210",
    "aod550": "207.210",
    "tcwv": "137.128"
}

server = ECMWFDataServer()

if __name__ == '__main__':
    logger.debug('begin ecmwf download ...')
    for year in xrange(2003, 2013):
        threads = []
        for name, param in params.iteritems():
            targetname = "%s_%s_macc.nc" % (name, year)
            if os.path.exists(targetname):
                 continue
            startdate = datetime(year, 1, 1).strftime('%Y-%m-%d')
            stopdate = datetime(year, 12, 31).strftime('%Y-%m-%d')
            logger.debug('begin downloading target: %s ...', targetname)
            t = threading.Thread(target=ecmwf, name=name, args=(param, targetname, startdate, stopdate))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            logger.debug('... target: %s download complete', t.name)
    logger.debug('... all downloads complete')
