#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 00:42:17 2018

@author: aravind

pytest -v

py.test -v

pytest -v -rxs

python -m pytest -v

pytest -k multiply 
"""

import funclib as f
import pytest
import sys 

@pytest.mark.skip(reason="Too lazy to run add test!")

def test_add():
    sum = f.add(4,-4)
    assert sum == 1

def test_multiply():
    mul = f.multiply(4,-4)
    assert mul == -16

@pytest.mark.skipif(sys.version_info > (2,7), reason="// Divide only works on py 3!")

def test_divide():
    div = f.divide(4,-4)
    assert div == -1

def test_divide_lp():
	div = f.divide_lp(4,-4)
	assert div == -1 


def test_subtract():
    sub = f.subtract(4,-4)
    assert sub == 8

    