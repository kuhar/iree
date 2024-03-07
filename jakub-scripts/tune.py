#!/usr/bin/env python3

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import argparse
import logging
import re
import z3
from dataclasses import asdict, dataclass
from os import path

logging.basicConfig(level=logging.DEBUG)

def create_template(filename):
    template = ''
    with open(filename, 'r') as f:
        template = f.readlines()
    return template

def apply_params(template, configuration):
    params = asdict(configuration)
    keys = list(params.keys())
    expr = re.compile(r'intrinsic = #iree_gpu.mfma_layout<([0-9A-Za-z_]+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+), subgroup_m_tile_count = ([0-9]+), subgroup_n_tile_count = ([0-9]+), subgroup_k_tile_count = ([0-9]+)>}>, workgroup_size = \[([0-9]+) : index, ([0-9]+) : index, ([0-9]+) : index\]')
    expr2 = re.compile(r'tile_sizes = \[\[([0-9]+), ([0-9]+), ([0-9]+)\]\]')
    repl = ''
    for i, key in enumerate(keys):
        if 'workgroup_size' in key or 'tile_sizes' in key or 'subgroup_size' in key:
            continue
        if len(repl) != 0:
            repl += ', '
        repl += f'{key} = {params[key]}'

    repl += '>}>, workgroup_size = ['
    for i, val in enumerate(params['workgroup_size']):
        repl += f'{val} : index'
        if i != len(params['workgroup_size']) - 1:
            repl += ', '
    repl += ']'

    repl2 = 'tile_sizes = [['
    for i, tile_size in enumerate(params['tile_sizes']):
        repl2 += f'{tile_size}'
        if i != len(params['tile_sizes']) - 1:
            repl2 += ', '
    repl2 += ']]'

    print("Repl: ", repl)
    print("Repl2: ", repl2)
    modified = ''
    for line in template:
        if 'intrinsic' in line:
            line = re.sub(expr, repl, line)
        if 'tile_sizes' in line:
            line = re.sub(expr2, repl2, line)
        modified += line

    return modified

def get_shapes(template):
    for line in template:
        if 'linalg.generic' not in line:
            continue
        if r'iterator_types = ["parallel", "parallel", "reduction"]' not in line:
            continue
        # ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>)
        shape = re.search(r'ins\(.+tensor<([0-9]+)x([0-9]+)xf16>, tensor<([0-9]+)x([0-9]+)xf16>\).+outs', line)
        if shape is None:
            continue

        assert len(shape.groups()) == 4
        M, K0, K1, N = shape.groups()
        assert K0 == K1
        return int(M), int(N), int(K0)

    assert False, "Shape not found"


def And(a, b):
    a_module = a.__class__.__module__
    b_module = b.__class__.__module__
    if a_module == b_module == 'builtins':
        return a and b
    return z3.And(a, b)

def is_pow2(x, min, max):
    return z3.Or(list(x == 2 ** i for i in range(min, max + 1)))

def generate_constraints(tile_sizes, subgroup_size, subgroup_m_count, subgroup_n_count,
                         subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
    m, n, k = tile_sizes
    workgroup_size = subgroup_size * 4
    constraints = [subgroup_size == 64]
    # constraints += [m >= 16, n >= 16, k >= 16]
    # constraints += [m <= 512, m <= 512, k <= 512]
    constraints += [is_pow2(m, 4, 9)]
    constraints += [is_pow2(n, 4, 9)]
    constraints += [is_pow2(k, 4, 9)]
    for x in (subgroup_m_count, subgroup_n_count,
              subgroup_n_count, subgroup_n_tile_count, subgroup_k_tile_count):
        constraints += [x >= 1, x <= 16]
        constraints += is_pow2(x, 0, 6)
    constraints += [m == subgroup_m_count * subgroup_m_tile_count * subgroup_size]
    constraints += [n == subgroup_n_count * subgroup_n_tile_count * subgroup_size]
    constraints += [k * subgroup_k_tile_count == workgroup_size]
    return constraints

@dataclass
class Configuration:
    subgroup_size : int
    workgroup_size : list[int]
    intrinsic : str
    tile_sizes : list[int]
    subgroup_m_count : int
    subgroup_n_count : int
    subgroup_m_tile_count : int
    subgroup_n_tile_count : int
    subgroup_k_tile_count : int

def generate_candidate(tile_sizes, M, N, K):
    m, n, k = tile_sizes
    subgroup_size = 64
    workgroup_size = 4 * subgroup_size
    subgroup_m_count = 2
    subgroup_n_count = 2
    # breakpoint()
    candidate = Configuration(64, [128, 2, 1], '#iree_gpu.mfma_layout<F16_16x16x16_F32>', tile_sizes,
                              subgroup_m_count, subgroup_n_count,
                              subgroup_m_tile_count=(m * subgroup_m_count) // subgroup_size,
                              subgroup_n_tile_count=(n * subgroup_n_count) // subgroup_size,
                              subgroup_k_tile_count=(k * workgroup_size // subgroup_size) // subgroup_size)
    return candidate


def generate_solutions():
    subgroup_size = z3.Int('subgroup_size')
    m, n, k = z3.Int('m'), z3.Int('n'), z3.Int('k')
    sg_m_cnt = z3.Int('sg_m_cnt')
    sg_n_cnt = z3.Int('sg_n_cnt')
    sg_m_tcnt = z3.Int('sg_m_tcnt')
    sg_n_tcnt = z3.Int('sg_n_tcnt')
    sg_k_tcnt = z3.Int('sg_k_tcnt')
    all_vars = [m, n, k, sg_m_cnt, sg_n_cnt, sg_m_tcnt, sg_n_tcnt, sg_k_tcnt]

    solver = z3.Solver()
    constraints = generate_constraints([m, n, k], subgroup_size, sg_m_cnt, sg_n_cnt, sg_m_tcnt, sg_n_tcnt, sg_k_tcnt)
    solver.add(z3.simplify(z3.And(constraints)))
    logging.debug(f'Initial constraints: {solver}')
    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()

        config = Configuration(lookup(subgroup_size), [128, 2, 1], '#iree_gpu.mfma_layout<F16_16x16x16_F32>',
                               [lookup(m), lookup(n), lookup(k)], lookup(sg_m_cnt), lookup(sg_n_cnt),
                               lookup(sg_m_tcnt), lookup(sg_n_tcnt), lookup(sg_k_tcnt))
        print(f'Solution #{i}: {config}')
        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1
        yield config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input mlir file', type=str)
    parser.add_argument('-o', '--output', help='Output dir', type=str)
    parser.add_argument('-l', '--limit', help='Max number of candidates generated', type=int, default=10)

    args = parser.parse_args()

    input_file = str(args.input)
    logging.debug(f'Processing {input_file}')
    template = create_template(input_file)
    M, N, K = get_shapes(template)

    for i, config in enumerate(generate_solutions())[:args.limit]:
        #params = generate_candidate(config.tile_sizes, M, N, K)
        new_mlir = apply_params(template, config)

        with open(path.join(args.output, f'candidate_{i}.mlir') , 'w') as f:
            f.write(new_mlir)
