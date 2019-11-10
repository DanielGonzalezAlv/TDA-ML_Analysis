#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze
"""

from __future__ import print_function

import os
import csv
import random
import time
from multiprocessing import Process, Value, Lock, Manager, Pool
from joblib import Parallel, delayed


# http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing

class Counter(object):
    def __init__(self, manager, initval=0):
        self.val = manager.Value('i', initval)
        self.lock = manager.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


def find_path(start, end, result, counter, total):
    for _ in range(50):
        time.sleep(random.random() / 10.0)
        counter.increment()
    result[i] = counter.value()


def get_nodes():
    nodes = []
    with open('C:/data/entity_resolution/redmart/output.tsv', 'rb') as fi:
        reader = csv.DictReader(fi)
        for row in reader:
            nodes.append(row['RegNum'])
    return nodes
