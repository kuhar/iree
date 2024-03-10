#! /usr/bin/env python3

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+')
args = parser.parse_args()
inputs = args.input
assert len(inputs) == 2
expected_f, actual_f = inputs

with open(expected_f, 'rb') as f:
  expected_output = np.load(f)

with open(actual_f, 'rb') as f:
  output = np.load(f)

error = np.max(np.abs(output - expected_output))
print("Max error = ", error)
