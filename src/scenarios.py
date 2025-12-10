import numpy as np
from PIL import Image
from scipy.special import erf
from src.fluid_solver import Fluid

import taichi as ti
import numpy as np
from PIL import Image

class StarryWaterScenario:
    DEFAULT_RESOLUTION = (512, 512)
    OUTPUT_FILENAME = "starry_water_simulation.gif"

    def __init__(self, duration=300):
        self.duration = duration
        self.resolution = self.DEFAULT_RESOLUTION
        self.h, self.w = self.resolution
        
        self.fluid = Fluid(
            self.resolution,
            'dye',
            viscosity=0.002,
            diffusion=0.001,
            dissipation=0.0005
        )
        
        self.canvas_tex = self._make_canvas_texture(self.resolution, grain=0.15, scale=5.0)

    def _make_canvas_texture(self, resolution, grain, scale, seed=42):
        np.random.seed(seed)
        h, w = resolution
        
        base = np.random.normal(loc=0.5, scale=0.2, size=(h, w))
        
        small_h, small_w = max(2, int(h/scale)), max(2, int(w/scale))
        small = Image.fromarray(np.clip(base * 255, 0, 255).astype('uint8'))
        small = small.resize((small_w, small_h), resample=Image.BILINEAR)
        big = small.resize((w, h), resample=Image.BICUBIC)
        
        arr = np.array(big).astype(np.float32) / 255.0
        
        texture = (1 - grain) * 0.5 + grain * arr
        texture = texture * 0.4 + 0.8 
        return texture

    def _apply_palette(self, curl_field, dye_field):
        color_bg_deep = np.array([0.05, 0.08, 0.25])
        color_bg_light = np.array([0.25, 0.45, 0.65])
        color_star_core = np.array([1.0, 0.85, 0.2])
        color_star_edge = np.array([0.9, 0.6, 0.1])

        mix_factor = (erf(curl_field * 2.0) + 1) / 2.0  
        mix_factor = np.clip(mix_factor, 0, 1)
        
        bg_color = (1 - mix_factor)[..., None] * color_bg_deep + mix_factor[..., None] * color_bg_light

        star_intensity = np.clip(dye_field, 0, 1)[..., None]
        star_color = (star_intensity ** 0.5) * color_star_core + (1 - star_intensity**0.5) * color_star_edge
        final_color = bg_color * (1 - star_intensity * 0.8) + star_color * star_intensity
        return np.clip(final_color * 255, 0, 255).astype('uint8')

    def _add_vortex(self, x, y, strength, radius):
        """
        Injects a Gaussian vortex into the velocity field.
        """
        _, H, W = self.fluid.velocity.shape
        yy, xx = np.mgrid[0:H, 0:W]
        
        dx = xx - x
        dy = yy - y
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq) + 1e-6
        
        falloff = np.exp(-dist_sq / (2 * radius**2))

        u = -dy / dist * strength * falloff
        v =  dx / dist * strength * falloff
        
        self.fluid.velocity[0] += u
        self.fluid.velocity[1] += v

    def step(self):
        print(f"Scenario: Starry Water | Resolution: {self.resolution} | Frames: {self.duration}")
        self._add_vortex(x=self.w*0.3, y=self.h*0.5, strength=80.0, radius=80)
        self._add_vortex(x=self.w*0.7, y=self.h*0.4, strength=-70.0, radius=90)
        
        # Add a gentle base flow at the bottom
        self.fluid.velocity[0, int(self.h*0.8):, :] += 1.0 

        num_stars = 12
        star_positions = []
        for _ in range(num_stars):
            sx = np.random.randint(20, self.w-20)
            sy = np.random.randint(20, int(self.h*0.6))
            star_positions.append((sx, sy))
            # Add small initial rotation to stars
            self._add_vortex(sx, sy, strength=15.0 * (1 if np.random.rand()>0.5 else -1), radius=15)

        frames = []

        for f in range(self.duration):
            if f % 20 == 0:
                print(f"  Rendering frame {f}/{self.duration}...")

            for (sx, sy) in star_positions:
                radius = 6
                y_s, y_e = max(0, sy-radius), min(self.h, sy+radius)
                x_s, x_e = max(0, sx-radius), min(self.w, sx+radius)
                
                y_slice, x_slice = np.ogrid[y_s:y_e, x_s:x_e]
                mask = (x_slice - sx)**2 + (y_slice - sy)**2 <= radius**2
                
                # Stop injection after 150 frames to let it drift
                if f < 150: 
                    intensity = 1.0 + 0.2 * np.sin(f * 0.2)
                    self.fluid.dye[y_s:y_e, x_s:x_e][mask] += 0.5 * intensity

            step_result = self.fluid.step()
            
            if isinstance(step_result, (tuple, list)) and len(step_result) >= 2:
                curl = step_result[1]
            else:
                curl = np.zeros((self.h, self.w)) 
                
            rgb = self._apply_palette(curl, self.fluid.dye)

            rgb = rgb * self.canvas_tex[..., None]
            rgb = np.clip(rgb, 0, 255).astype('uint8')
            
            frames.append(Image.fromarray(rgb))
            
        return frames
    
    def setup_gui(self):
        self.fluid = Fluid(
            self.resolution,
            'dye',
            viscosity=0.002,
            diffusion=0.001,
            dissipation=0.0005
        )

        self._add_vortex(x=self.w*0.3, y=self.h*0.5, strength=80.0, radius=80)
        self._add_vortex(x=self.w*0.7, y=self.h*0.4, strength=-70.0, radius=90)
        self.fluid.velocity[0, int(self.h*0.8):, :] += 1.0 

        self.gui_star_positions = []
        num_stars = 12
        for _ in range(num_stars):
            sx = np.random.randint(20, self.w-20)
            sy = np.random.randint(20, int(self.h*0.6))
            self.gui_star_positions.append((sx, sy))
            self._add_vortex(sx, sy, strength=15.0 * (1 if np.random.rand()>0.5 else -1), radius=15)

    def render_gui_frame(self, frame_idx):
        f = frame_idx
        
        for (sx, sy) in self.gui_star_positions:
            radius = 6
            y_s, y_e = max(0, sy-radius), min(self.h, sy+radius)
            x_s, x_e = max(0, sx-radius), min(self.w, sx+radius)
            
            y_slice, x_slice = np.ogrid[y_s:y_e, x_s:x_e]
            mask = (x_slice - sx)**2 + (y_slice - sy)**2 <= radius**2
            
            if f < 150: 
                intensity = 1.0 + 0.2 * np.sin(f * 0.2)
                self.fluid.dye[y_s:y_e, x_s:x_e][mask] += 0.5 * intensity

        step_result = self.fluid.step()
        
        if isinstance(step_result, (tuple, list)) and len(step_result) >= 2:
            curl = step_result[1]
        else:
            curl = np.zeros((self.h, self.w))

        rgb = self._apply_palette(curl, self.fluid.dye)
        rgb = rgb * self.canvas_tex[..., None]
        return np.clip(rgb, 0, 255).astype('uint8')


import taichi as ti
from src.fluid_solver import Fluid3D
from src.utils import sample_trilinear

@ti.data_oriented
class Quicksand:
    def __init__(self, duration=1000):
        self.fluid = Fluid3D(jacobi_iters=duration)

        self.pn_max = 10000000
        self.pn_current = 0
        self.rate = 100000

        self.particles = ti.Vector.field(3, dtype=ti.f32, shape=self.pn_max)
        self.particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.pn_max)
        self.particle_radius = 0.0015

        self.source_center_L = ti.Vector([0.05, 0.5, 0.5])
        self.source_center_R = ti.Vector([0.95, 0.5, 0.5])
        self.source_velocity_L = ti.Vector([0.01, 0.0, 0.0])
        self.source_velocity_R = ti.Vector([-0.01, 0.0, 0.0])
        self.source_radius = 0.001

        self.init_particles()

    @ti.kernel
    def init_particles(self):
        for i in range(self.pn_max):
            r = ti.sqrt(ti.random()) * self.source_radius
            a = ti.random() * 6.28318
            b = ti.random() * 3.141592

            offset = ti.Vector([
                r * ti.cos(a) * ti.sin(b),
                r * ti.sin(a) * ti.sin(b),
                r * ti.cos(b)
            ])

            if i % 2 == 0:
                self.particles[i] = offset + self.source_center_L
                self.particle_colors[i] = ti.Vector([0, 1, 1])
            else:
                self.particles[i] = offset + self.source_center_R
                self.particle_colors[i] = ti.Vector([1, 0.5, 0])

    @ti.kernel
    def update_particles(self, pn: ti.i32):
        dt = self.fluid.dt
        for i in range(pn):
            v = sample_trilinear(self.fluid.velocities, self.particles[i])
            self.particles[i] += v * dt

    def step(self):
        fluid = self.fluid

        fluid.advect()

        fluid.add_force(self.source_center_L, self.source_radius, self.source_velocity_L)
        fluid.add_force(self.source_center_R, self.source_radius, self.source_velocity_R)

        fluid.solve_divergence()
        fluid.pressure_solve()
        fluid.project()

        if self.pn_current < self.pn_max:
            self.pn_current += self.rate

        self.update_particles(self.pn_current)
    

SCENARIO_MAP = {
    'starry_water': StarryWaterScenario,
    'quick_sand': Quicksand
}