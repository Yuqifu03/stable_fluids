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
from src.utils import sample_trilinear,semi_lagrangian

@ti.data_oriented
class Fluid3D:
    def __init__(self, n=128, dt=0.03, rho=1, jacobi_iters=30,
                 RK=3, 
                 enable_clipping=True):
        self.n = n
        self.dt = dt
        self.dx = 1 / n
        self.rho = rho
        self.jacobi_iters = jacobi_iters
        self.RK = RK
        self.enable_clipping = enable_clipping

        self.stagger = ti.Vector([0.5, 0.5, 0.5])

        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
        self.new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
        
        self.pressures = ti.field(dtype=ti.f32, shape=(n, n, n))
        self.new_pressures = ti.field(dtype=ti.f32, shape=(n, n, n))
        self.divergences = ti.field(dtype=ti.f32, shape=(n, n, n))
        self.curl = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))

        self._build_kernels()

    @ti.kernel
    def init_velocity_field(self):
        for i in ti.grouped(self.velocities):
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])

    def _build_kernels(self):
        # 1. Advect
        @ti.kernel
        def advect_kernel(vel: ti.template(), field: ti.template(), new_field: ti.template(), dt: ti.f32):
            semi_lagrangian(vel, field, new_field, dt)

        # 3. Divergence
        @ti.kernel
        def solve_div_kernel(vel: ti.template(), divergences: ti.template()):
            dx = self.dx
            stagger = self.stagger
            n = self.n
            for i, j, k in vel:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                v_l = sample_trilinear(vel, c - ti.Vector([1, 0, 0]) * dx).x
                v_r = sample_trilinear(vel, c + ti.Vector([1, 0, 0]) * dx).x
                v_d = sample_trilinear(vel, c - ti.Vector([0, 1, 0]) * dx).y
                v_u = sample_trilinear(vel, c + ti.Vector([0, 1, 0]) * dx).y
                v_b = sample_trilinear(vel, c - ti.Vector([0, 0, 1]) * dx).z
                v_f = sample_trilinear(vel, c + ti.Vector([0, 0, 1]) * dx).z
                
                # 简单边界
                if i == 0: v_l = -sample_trilinear(vel, c).x
                if i == n - 1: v_r = -sample_trilinear(vel, c).x
                if j == 0: v_d = -sample_trilinear(vel, c).y
                if j == n - 1: v_u = -sample_trilinear(vel, c).y
                if k == 0: v_b = -sample_trilinear(vel, c).z
                if k == n - 1: v_f = -sample_trilinear(vel, c).z

                divergences[i, j, k] = (v_r - v_l + v_u - v_d + v_f - v_b) / (2 * dx)

        # 4. Pressure Jacobi
        @ti.kernel
        def jacobi_pressure_kernel(pressures: ti.template(), new_pressures: ti.template()):
            rho = self.rho
            dt = self.dt
            dx = self.dx
            stagger = self.stagger
            for i, j, k in pressures:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                s = sample_trilinear(pressures, c + ti.Vector([1, 0, 0]) * dx) + \
                    sample_trilinear(pressures, c - ti.Vector([1, 0, 0]) * dx) + \
                    sample_trilinear(pressures, c + ti.Vector([0, 1, 0]) * dx) + \
                    sample_trilinear(pressures, c - ti.Vector([0, 1, 0]) * dx) + \
                    sample_trilinear(pressures, c + ti.Vector([0, 0, 1]) * dx) + \
                    sample_trilinear(pressures, c - ti.Vector([0, 0, 1]) * dx)
                new_pressures[i, j, k] = (s - self.divergences[i, j, k] * rho / dt * dx * dx) / 6.0

        # 5. Project
        @ti.kernel
        def project_kernel(vel: ti.template(), pressures: ti.template()):
            rho = self.rho
            dt = self.dt
            dx = self.dx
            stagger = self.stagger
            for i, j, k in vel:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                p_r = sample_trilinear(pressures, c + ti.Vector([1, 0, 0]) * dx)
                p_l = sample_trilinear(pressures, c - ti.Vector([1, 0, 0]) * dx)
                p_u = sample_trilinear(pressures, c + ti.Vector([0, 1, 0]) * dx)
                p_d = sample_trilinear(pressures, c - ti.Vector([0, 1, 0]) * dx)
                p_f = sample_trilinear(pressures, c + ti.Vector([0, 0, 1]) * dx)
                p_b = sample_trilinear(pressures, c - ti.Vector([0, 0, 1]) * dx)
                grad_p = ti.Vector([p_r - p_l, p_u - p_d, p_f - p_b]) / (2 * dx)
                vel[i, j, k] -= grad_p / rho * dt

        # 6. Force
        @ti.kernel
        def apply_force_kernel(vel: ti.template(), pos: ti.types.vector(3, ti.f32), r: ti.f32, force: ti.types.vector(3, ti.f32)):
            dt = self.dt
            dx = self.dx
            stagger = self.stagger
            dp = force + (ti.Vector([ti.random(), ti.random(), ti.random()]) - 0.5) * 0.0001
            for i, j, k in vel:
                p = ti.Vector([(i + stagger.x) * dx, (j + stagger.y) * dx, (k + stagger.z) * dx])
                d2 = (p - pos).norm_sqr()
                vel[i, j, k] += dp * dt * ti.exp(-d2 / (0.2 * r)) * 40

        self.advect_kernel = advect_kernel
        self.solve_div_kernel = solve_div_kernel
        self.jacobi_pressure_kernel = jacobi_pressure_kernel
        self.project_kernel = project_kernel
        self.apply_force_kernel = apply_force_kernel

        # ------------- 7. Compute Curl----------------
        @ti.kernel
        def compute_curl_kernel(vel: ti.template(), curl: ti.template()):
            dx = self.dx
            stagger = self.stagger
            for i, j, k in vel:
                c = (ti.Vector([i, j, k]) + stagger) * dx

                v_u = sample_trilinear(vel, c + ti.Vector([0, 1, 0]) * dx).z
                v_d = sample_trilinear(vel, c - ti.Vector([0, 1, 0]) * dx).z
                v_f = sample_trilinear(vel, c + ti.Vector([0, 0, 1]) * dx).y
                v_b = sample_trilinear(vel, c - ti.Vector([0, 0, 1]) * dx).y
                curl_x = (v_u - v_d) - (v_f - v_b)

                # du/dz - dw/dx
                v_f_x = sample_trilinear(vel, c + ti.Vector([0, 0, 1]) * dx).x
                v_b_x = sample_trilinear(vel, c - ti.Vector([0, 0, 1]) * dx).x
                v_r_z = sample_trilinear(vel, c + ti.Vector([1, 0, 0]) * dx).z
                v_l_z = sample_trilinear(vel, c - ti.Vector([1, 0, 0]) * dx).z
                curl_y = (v_f_x - v_b_x) - (v_r_z - v_l_z)

                # dv/dx - du/dy
                v_r_y = sample_trilinear(vel, c + ti.Vector([1, 0, 0]) * dx).y
                v_l_y = sample_trilinear(vel, c - ti.Vector([1, 0, 0]) * dx).y
                v_u_x = sample_trilinear(vel, c + ti.Vector([0, 1, 0]) * dx).x
                v_d_x = sample_trilinear(vel, c - ti.Vector([0, 1, 0]) * dx).x
                curl_z = (v_r_y - v_l_y) - (v_u_x - v_d_x)

                curl[i, j, k] = ti.Vector([curl_x, curl_y, curl_z]) / (2 * dx)

        @ti.kernel
        def vorticity_confinement_kernel(vel: ti.template(), curl: ti.template(), strength: ti.f32):
            dt = self.dt
            dx = self.dx
            stagger = self.stagger
            
            for i, j, k in vel:
                c = (ti.Vector([i, j, k]) + stagger) * dx
                
                curl_len_r = sample_trilinear(curl, c + ti.Vector([1, 0, 0]) * dx).norm()
                curl_len_l = sample_trilinear(curl, c - ti.Vector([1, 0, 0]) * dx).norm()
                curl_len_u = sample_trilinear(curl, c + ti.Vector([0, 1, 0]) * dx).norm()
                curl_len_d = sample_trilinear(curl, c - ti.Vector([0, 1, 0]) * dx).norm()
                curl_len_f = sample_trilinear(curl, c + ti.Vector([0, 0, 1]) * dx).norm()
                curl_len_b = sample_trilinear(curl, c - ti.Vector([0, 0, 1]) * dx).norm()

                grad_len = ti.Vector([
                    curl_len_r - curl_len_l,
                    curl_len_u - curl_len_d,
                    curl_len_f - curl_len_b
                ]) / (2 * dx)

                grad_len_norm = grad_len.norm()

                if grad_len_norm > 1e-5:
                    N = grad_len / grad_len_norm
                    omega = curl[i, j, k]
                    force = N.cross(omega) * strength * dx
                    
                    vel[i, j, k] += force * dt
        
        self.compute_curl_kernel = compute_curl_kernel
        self.vorticity_confinement_kernel = vorticity_confinement_kernel

    # ------------------ High-level API ------------------
    def vorticity_confinement(self, strength=2.0):
        self.compute_curl_kernel(self.velocities, self.curl)
        self.vorticity_confinement_kernel(self.velocities, self.curl, strength)

    def advect(self):
        self.advect_kernel(self.velocities, self.velocities, self.new_velocities, self.dt)
        self.velocities, self.new_velocities = self.new_velocities, self.velocities

    def add_force(self, pos, r, force):
        self.apply_force_kernel(self.velocities, pos, r, force)

    def project(self):
        """Standard projection step: Divergence -> Solve Pressure -> Subtract Gradient"""
        self.solve_div_kernel(self.velocities, self.divergences)
        for _ in range(self.jacobi_iters):
            self.jacobi_pressure_kernel(self.pressures, self.new_pressures)
            self.pressures, self.new_pressures = self.new_pressures, self.pressures
        self.project_kernel(self.velocities, self.pressures)