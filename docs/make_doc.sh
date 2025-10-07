#!/bin/bash

export PYTHONPATH=../src/python/epiclass

# from git root
pdoc3 --html -o . ../src/python/epiclass --force
