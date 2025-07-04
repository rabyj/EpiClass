#!/bin/bash

export PYTHONPATH=../src/python/epi_ml

# from git root
pdoc3 --html -o . ../src/python/epi_ml --force
