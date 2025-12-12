from functools import reduce
from itertools import cycle
from math import factorial

import numpy as np
import scipy.sparse as sp

def difference(derivative, accuracy=1):
    derivative += 1
    radius = accuracy + derivative // 2 - 1
    points = range(-radius, radius + 1)
    coefficients = np.linalg.inv(np.vander(points))
    return coefficients[-derivative] * factorial(derivative - 1), list(points)


def operator(shape, difference_stencil, boundary='periodic'):
    coeffs, offsets = difference_stencil
    def build_1d_operator(dim):
        data = [np.full(dim, c) for c in coeffs]
        diags = list(offsets)
        
        if boundary == 'periodic':
            for i, offset in enumerate(offsets):
                if offset < 0:
                    diags.append(dim + offset)
                    data.append(np.full(dim, coeffs[i]))
                elif offset > 0:
                    diags.append(-(dim - offset)) 
                    data.append(np.full(dim, coeffs[i]))
        return sp.diags(data, diags, shape=(dim, dim), format='csc')

    factors = [build_1d_operator(dim) for dim in shape]
   
    return reduce(lambda a, f: sp.kronsum(f, a, format='csc'), factors)

import taichi as ti
import numpy as np

@ti.func
def I(i, j, k):
    return ti.Vector([i, j, k])

@ti.func
def clamp(p):
    for d in ti.static(range(3)):
        p[d] = min(1 - 1/128 * 0.5 - 1e-4, max(p[d], 1/128 * 0.5))
    return p

@ti.func
def sample_trilinear(field, p):
    n = field.shape[0]
    dx = 1 / n
    stagger = ti.Vector([0.5, 0.5, 0.5])
    p = clamp(p)
    grid_f = p * n - stagger
    grid_i = ti.cast(ti.floor(grid_f), ti.i32)
    d = grid_f - grid_i
    return (
        field[grid_i] * (1 - d.x) * (1 - d.y) * (1 - d.z)
        + field[grid_i + I(1, 0, 0)] * d.x * (1 - d.y) * (1 - d.z)
        + field[grid_i + I(0, 1, 0)] * (1 - d.x) * d.y * (1 - d.z)
        + field[grid_i + I(1, 1, 0)] * d.x * d.y * (1 - d.z)
        + field[grid_i + I(0, 0, 1)] * (1 - d.x) * (1 - d.y) * d.z
        + field[grid_i + I(1, 0, 1)] * d.x * (1 - d.y) * d.z
        + field[grid_i + I(0, 1, 1)] * (1 - d.x) * d.y * d.z
        + field[grid_i + I(1, 1, 1)] * d.x * d.y * d.z
    )

@ti.func
def sample_min(field, p):
    n = field.shape[0]
    dx = 1 / n
    stagger = ti.Vector([0.5, 0.5, 0.5])
    p = clamp(p)
    grid_f = p * n - stagger
    grid_i = ti.cast(ti.floor(grid_f), ti.i32)
    return min(
        field[grid_i], field[grid_i + I(1,0,0)], field[grid_i + I(0,1,0)], field[grid_i + I(1,1,0)],
        field[grid_i + I(0,0,1)], field[grid_i + I(1,0,1)], field[grid_i + I(0,1,1)], field[grid_i + I(1,1,1)]
    )

@ti.func
def sample_max(field, p):
    n = field.shape[0]
    dx = 1 / n
    stagger = ti.Vector([0.5, 0.5, 0.5])
    p = clamp(p)
    grid_f = p * n - stagger
    grid_i = ti.cast(ti.floor(grid_f), ti.i32)
    return max(
        field[grid_i], field[grid_i + I(1,0,0)], field[grid_i + I(0,1,0)], field[grid_i + I(1,1,0)],
        field[grid_i + I(0,0,1)], field[grid_i + I(1,0,1)], field[grid_i + I(0,1,1)], field[grid_i + I(1,1,1)]
    )

@ti.func
def backtrace(vel, p, dt):
    v0 = sample_trilinear(vel, p)
    p1 = p - v0 * dt * 0.5
    v1 = sample_trilinear(vel, p1)
    p2 = p - v1 * dt * 0.75
    v2 = sample_trilinear(vel, p2)
    return p - (2/9*v0 + 1/3*v1 + 4/9*v2) * dt

@ti.func
def semi_lagrangian(vel, field, new_field, dt):
    n = field.shape[0]
    dx = 1 / n
    stagger = ti.Vector([0.5, 0.5, 0.5])
    for i in ti.grouped(field):
        p = (i + stagger) * dx
        new_field[i] = sample_trilinear(field, backtrace(vel, p, dt))
