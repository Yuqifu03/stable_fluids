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
        self.fluid = Fluid3D(
            dt=0.01,           
            jacobi_iters=30,
        )

        self.pn_max = 800000 
        self.pn_current = 0
        self.rate = 8000      
        
        self.frame_count = 0
        self.emit_end_frame = 50 

        self.particles = ti.Vector.field(3, dtype=ti.f32, shape=self.pn_max)
        self.particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.pn_max)
        
        self.particle_radius = 0.0005

        self.source_center = ti.Vector([0.5, 0.9, 0.5]) 
        
        self.emit_radius = 0.006 
        self.force_radius = 0.05
        
        self.source_velocity = ti.Vector([0.0, -1.0, 0.0])

        self.init_particles()

    @ti.kernel
    def init_particles(self):
        for i in range(self.pn_max):
            self.particles[i] = ti.Vector([-1.0, -1.0, -1.0])

    @ti.kernel
    def emit_particles(self, start_idx: ti.i32, end_idx: ti.i32, time: ti.f32):
        for i in range(start_idx, end_idx):
            r = ti.pow(ti.random(), 1.0/2.0) * self.emit_radius
            angle = ti.random() * 6.2831853

            smear_length = 0.03
            random_offset_y = (ti.random() - 0.5) * smear_length

            offset = ti.Vector([
                r * ti.cos(angle),
                random_offset_y, 
                r * ti.sin(angle)
            ])

            self.particles[i] = self.source_center + offset
            
            t = time * 0.1
            r_val = 0.5 + 0.5 * ti.sin(t)        
            g_val = 0.5 + 0.5 * ti.cos(t * 0.7)  
            b_val = 0.9                          
            
            self.particle_colors[i] = ti.Vector([r_val, g_val, b_val])

    @ti.kernel
    def update_particles(self, pn: ti.i32):
        dt = self.fluid.dt
        for i in range(pn):
            p = self.particles[i]
            if p.x > 0.02 and p.x < 0.98 and p.y > 0.02 and p.y < 0.98 and p.z > 0.02 and p.z < 0.98:
                v = sample_trilinear(self.fluid.velocities, p)
                self.particles[i] += v * dt
            else:
                self.particles[i] = ti.Vector([-1.0, -1.0, -1.0])

    def step(self):
        fluid = self.fluid
        
        fluid.advect()

        if self.frame_count < self.emit_end_frame:

            wobble_x = np.sin(self.frame_count * 0.2) * 0.15
            wobble_z = np.cos(self.frame_count * 0.15) * 0.15

            noise_x = (np.random.rand() - 0.5) * 0.2
            noise_z = (np.random.rand() - 0.5) * 0.2

            force_dir = ti.Vector([wobble_x + noise_x, -0.6, wobble_z + noise_z]) 
            
            fluid.add_force(self.source_center, self.force_radius, force_dir)

            num_emit = 8000
            start_idx = self.pn_current % self.pn_max
            end_idx = start_idx + num_emit
            
            if self.pn_current < self.pn_max:
                self.pn_current += num_emit
            
            if end_idx > self.pn_max:
                self.emit_particles(start_idx, self.pn_max, float(self.frame_count))
                self.emit_particles(0, end_idx - self.pn_max, float(self.frame_count))
            else:
                self.emit_particles(start_idx, end_idx, float(self.frame_count))

        fluid.vorticity_confinement(strength=10.0)
        
        fluid.project()

        active_count = min(self.pn_current, self.pn_max)
        self.update_particles(active_count)
        
        self.frame_count += 1
 
SCENARIO_MAP = {
    'starry_water': StarryWaterScenario,
    'quick_sand': Quicksand
}