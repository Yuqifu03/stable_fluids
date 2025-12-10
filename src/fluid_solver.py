import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import factorized, cg
from scipy.sparse import eye

from .utils import difference, operator

class Fluid:
    def __init__(self, shape, *quantities, viscosity=0.001, diffusion=0.001, dissipation=0.001, pressure_order=1):
        self.shape = shape
        self.dimensions = len(shape)
        self.viscosity = viscosity
        self.diffusion = diffusion
        self.dissipation = dissipation
        
        self.quantities = quantities
        for q in quantities:
            setattr(self, q, np.zeros(shape))

        self.indices = np.indices(shape)
        self.velocity = np.zeros((self.dimensions, *shape))

        self.laplacian_matrix = operator(shape, difference(2, pressure_order))
        if self.dimensions >= 3:
            def cg_solver(A):
                def solve(b):
                    x, info = cg(A, b, rtol=1e-5, maxiter=50)
                    return x
                return solve
            self.pressure_solver = cg_solver(self.laplacian_matrix)
        else:
            self.pressure_solver = factorized(self.laplacian_matrix)
        self.diffuse_velocity_solver = None 
        self.diffuse_scalar_solver = None

    def build_diffusion_solvers(self, dt):
        I = eye(self.laplacian_matrix.shape[0])
        
        diff_op_v = I - (self.viscosity * dt) * self.laplacian_matrix
        self.diffuse_velocity_solver = factorized(diff_op_v)
        
        diff_op_s = I - (self.diffusion * dt) * self.laplacian_matrix
        self.diffuse_scalar_solver = factorized(diff_op_s)

    def advect_field(self, field, vector_field, dt):
        advection_map = self.indices - vector_field * dt
        
        return map_coordinates(field, advection_map, prefilter=False, order=1, mode='wrap')

    def diffuse_field(self, field, solver):
        flat_field = field.flatten()
        return solver(flat_field).reshape(self.shape)

    def _gradient(self, field, axis):
        return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / 2
    
    def step(self, dt=1.0, forces=None):
        if self.diffuse_velocity_solver is None:
            self.build_diffusion_solvers(dt)

        # --- 1. ADD FORCES ---
        if forces is not None:
            self.velocity += forces * dt

        # --- 2. ADVECT ---
        new_velocity = np.zeros_like(self.velocity)
        for d in range(self.dimensions):
            new_velocity[d] = self.advect_field(self.velocity[d], self.velocity, dt)
        self.velocity = new_velocity

        for q in self.quantities:
            current_q = getattr(self, q)
            setattr(self, q, self.advect_field(current_q, self.velocity, dt))

        # --- 3. DIFFUSE ---
        for d in range(self.dimensions):
            self.velocity[d] = self.diffuse_field(self.velocity[d], self.diffuse_velocity_solver)

        for q in self.quantities:
            current_q = getattr(self, q)
            setattr(self, q, self.diffuse_field(current_q, self.diffuse_scalar_solver))

        # --- 4. PROJECT ---
        # Divergence
        divergence = np.zeros(self.shape)
        for d in range(self.dimensions):
            divergence += self._gradient(self.velocity[d], axis=d)

        # Pressure
        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)

        # Subtract Gradient
        for d in range(self.dimensions):
            self.velocity[d] -= self._gradient(pressure, axis=d)
        
        if self.dimensions == 2:
            du_dy = self._gradient(self.velocity[1], axis=0)
            dv_dx = self._gradient(self.velocity[0], axis=1)
            curl = dv_dx - du_dy
        else:
            curl = np.zeros(self.shape)

        # --- 5. DISSIPATE SCALARS ---
        for q in self.quantities:
            current_q = getattr(self, q)
            setattr(self, q, current_q / (1 + dt * self.dissipation))

        return divergence, curl, pressure

import taichi as ti
import numpy as np
from src.utils import (
    I, clamp, sample_trilinear, sample_min, sample_max,
    backtrace, semi_lagrangian, BFECC
)

class Fluid3D:
    def __init__(self, n=128, dt=0.03, rho=1, jacobi_iters=100,
                 RK=3, enable_BFECC=True, enable_clipping=True):
        self.n = n
        self.dt = dt
        self.dx = 1 / n
        self.rho = rho
        self.jacobi_iters = jacobi_iters
        self.RK = RK
        self.enable_BFECC = enable_BFECC
        self.enable_clipping = enable_clipping

        self.stagger = ti.Vector([0.5, 0.5, 0.5])

        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
        self.new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
        self.new_new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))

        self.pressures = ti.field(dtype=ti.f32, shape=(n, n, n))
        self.new_pressures = ti.field(dtype=ti.f32, shape=(n, n, n))
        self.divergences = ti.field(dtype=ti.f32, shape=(n, n, n))

        self._build_kernels()

    @ti.kernel
    def init_velocity_field(self):
        for i in ti.grouped(self.velocities):
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])

    def _build_kernels(self):

        # ------------- 1. Advect kernel -----------------
        @ti.kernel
        def advect_kernel(
            vel: ti.template(),
            field: ti.template(),
            new_field: ti.template(),
            new_new_field: ti.template(),
            dt: ti.f32
        ):
            if ti.static(self.enable_BFECC):
                BFECC(vel, field, new_field, new_new_field, dt)
            else:
                semi_lagrangian(vel, field, new_field, dt)

        # ------------- 2. Divergence kernel --------------
        @ti.kernel
        def solve_div_kernel(
            vel: ti.template(),
            divergences: ti.template()
        ):
            n = self.n
            dx = self.dx
            stagger = self.stagger

            for i, j, k in vel:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                l = c - ti.Vector([1, 0, 0]) * dx
                r = c + ti.Vector([1, 0, 0]) * dx
                d = c - ti.Vector([0, 1, 0]) * dx
                u = c + ti.Vector([0, 1, 0]) * dx
                b = c - ti.Vector([0, 0, 1]) * dx
                f = c + ti.Vector([0, 0, 1]) * dx

                v_c = sample_trilinear(vel, c)
                v_l = sample_trilinear(vel, l).x
                v_r = sample_trilinear(vel, r).x
                v_d = sample_trilinear(vel, d).y
                v_u = sample_trilinear(vel, u).y
                v_b = sample_trilinear(vel, b).z
                v_f = sample_trilinear(vel, f).z

                if i == 0: v_l = -v_c.x
                if i == n - 1: v_r = -v_c.x
                if j == 0: v_d = -v_c.y
                if j == n - 1: v_u = -v_c.y
                if k == 0: v_b = -v_c.z
                if k == n - 1: v_f = -v_c.z

                divergences[i, j, k] = (v_r - v_l + v_u - v_d + v_f - v_b) / (2 * dx)

        # ------------- 3. Jacobi pressure solve -----------
        @ti.kernel
        def jacobi_kernel(
            pressures: ti.template(),
            new_pressures: ti.template()
        ):
            n = self.n
            dx = self.dx
            dt = self.dt
            rho = self.rho
            stagger = self.stagger

            for i, j, k in pressures:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                offsets = [
                    ti.Vector([1, 0, 0]), ti.Vector([-1, 0, 0]),
                    ti.Vector([0, 1, 0]), ti.Vector([0, -1, 0]),
                    ti.Vector([0, 0, 1]), ti.Vector([0, 0, -1]),
                ]
                s = 0.0
                for off in ti.static(offsets):
                    s += sample_trilinear(pressures, c + off * dx)

                new_pressures[i, j, k] = (s - self.divergences[i, j, k] * rho / dt * dx * dx) / 6

        # ------------- 4. Project velocity field -----------
        @ti.kernel
        def project_kernel(
            vel: ti.template(),
            pressures: ti.template()
        ):
            n = self.n
            dx = self.dx
            rho = self.rho
            dt = self.dt
            stagger = self.stagger

            for i, j, k in vel:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                p_r = sample_trilinear(pressures, c + ti.Vector([1, 0, 0]) * dx)
                p_l = sample_trilinear(pressures, c - ti.Vector([1, 0, 0]) * dx)
                p_u = sample_trilinear(pressures, c + ti.Vector([0, 1, 0]) * dx)
                p_d = sample_trilinear(pressures, c - ti.Vector([0, 1, 0]) * dx)
                p_f = sample_trilinear(pressures, c + ti.Vector([0, 0, 1]) * dx)
                p_b = sample_trilinear(pressures, c - ti.Vector([0, 0, 1]) * dx)

                grad_p = ti.Vector([
                    p_r - p_l,
                    p_u - p_d,
                    p_f - p_b
                ]) / (2 * dx)

                vel[i, j, k] -= grad_p / rho * dt

        # ------------- 5. Apply force kernel ----------------
        @ti.kernel
        def apply_force_kernel(
            vel: ti.template(),
            pos: ti.types.vector(3, ti.f32),
            r: ti.f32,
            force: ti.types.vector(3, ti.f32)
        ):
            dt = self.dt
            dx = self.dx
            stagger = self.stagger

            dp = force + (ti.Vector([ti.random(), ti.random(), ti.random()]) - 0.5) * 0.001

            for i, j, k in vel:
                p = ti.Vector([(i + stagger.x) * dx,
                               (j + stagger.y) * dx,
                               (k + stagger.z) * dx])   # ← 修复你的 typo
                d2 = (p - pos).norm_sqr()
                radius = 0.2 * r
                vel[i, j, k] += dp * dt * ti.exp(-d2 / radius) * 40

        self.advect_kernel = advect_kernel
        self.solve_div_kernel = solve_div_kernel
        self.jacobi_kernel = jacobi_kernel
        self.project_kernel = project_kernel
        self.apply_force_kernel = apply_force_kernel

    # ------------------ high-level API ------------------
    def advect(self):
        self.advect_kernel(
            self.velocities,
            self.velocities,
            self.new_velocities,
            self.new_new_velocities,
            self.dt
        )
        self.velocities, self.new_velocities = self.new_velocities, self.velocities

    def add_force(self, pos, r, force):
        self.apply_force_kernel(self.velocities, pos, r, force)

    def solve_divergence(self):
        self.solve_div_kernel(self.velocities, self.divergences)

    def pressure_solve(self):
        for _ in range(self.jacobi_iters):
            self.jacobi_kernel(self.pressures, self.new_pressures)
            self.pressures, self.new_pressures = self.new_pressures, self.pressures

    def project(self):
        self.project_kernel(self.velocities, self.pressures)
