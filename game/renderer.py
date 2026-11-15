from __future__ import annotations

import math
import numpy as np

from .surfer import SurferState
from .terrain import DT, PI, clamp01, surf_geometry, terrain_parameters

class WaveField:
    """Leapfrog 2-D wave solver. CFL limit uses documented 0.70710678 constant."""
    def __init__(self, size: int = 128, ratio: float = 0.45, damping: float = 0.001):
        if ratio > 0.7071067811865476:
            raise ValueError("CFL condition violated")
        self.size, self.ratio, self.damping = size, ratio, damping
        self.u = np.zeros((size, size), dtype=np.float32)
        self.old = np.zeros_like(self.u)
        self._next = np.zeros_like(self.u)
        self._lap = np.zeros_like(self.u)
        self._scratch = np.zeros_like(self.u)
        self.version = 0

    def step(self) -> None:
        u = self.u
        # External scientific tests may replace the public arrays with float64
        # modes. Rebuild once in that exceptional case; normal gameplay stays
        # on the preallocated float32 fast path.
        if self._next.shape != u.shape or self._next.dtype != u.dtype:
            self._next = np.zeros_like(u)
            self._lap = np.zeros_like(u)
            self._scratch = np.zeros_like(u)
        lap = self._lap
        inner = lap[1:-1, 1:-1]
        np.add(u[2:, 1:-1], u[:-2, 1:-1], out=inner)
        np.add(inner, u[1:-1, 2:], out=inner)
        np.add(inner, u[1:-1, :-2], out=inner)
        np.multiply(u[1:-1, 1:-1], np.float32(4.0), out=self._scratch[1:-1, 1:-1])
        np.subtract(inner, self._scratch[1:-1, 1:-1], out=inner)
        new = self._next
        np.multiply(lap, self.ratio * self.ratio, out=new)
        np.multiply(u, np.float32(2.0), out=self._scratch)
        np.add(new, self._scratch, out=new)
        np.subtract(new, self.old, out=new)
        np.subtract(u, self.old, out=self._scratch)
        np.multiply(self._scratch, self.damping, out=self._scratch)
        np.subtract(new, self._scratch, out=new)
        new[0, :] = new[-1, :] = 0.0
        new[:, 0] = new[:, -1] = 0.0
        self.old, self.u, self._next = u, new, self.old
        self.version += 1

    def inject(self, x: int, y: int, radius: int = 9, amplitude: float = 0.8) -> None:
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                q = (i * i + j * j) / float(radius * radius)
                if q <= 1.0:
                    xx, yy = x + i, y + j
                    if 0 < xx < self.size - 1 and 0 < yy < self.size - 1:
                        self.u[yy, xx] += amplitude * math.sin(PI * (1.0 - q)) ** 2
        self.version += 1

    def energy(self) -> float:
        velocity = self.u - self.old
        gx = self.u[:, 1:] - self.u[:, :-1]
        gy = self.u[1:, :] - self.u[:-1, :]
        return float(0.5 * (np.sum(velocity * velocity) + self.ratio * self.ratio * (np.sum(gx * gx) + np.sum(gy * gy))))



VERTEX_SHADER = """#version 330
in vec2 in_pos;
out vec2 uv;
void main(){ uv=in_pos*0.5+0.5; gl_Position=vec4(in_pos,0.0,1.0); }
"""


SURF_FRAGMENT_SHADER = """#version 330
uniform vec2 resolution;
uniform float time;
uniform float travel;
uniform float level_length;
uniform float surfer_y;
uniform float camera_y;
uniform float surfer_pitch;
uniform float surfer_vy;
uniform float speed;
uniform float fuel;
uniform float boost;
uniform float hurt;
uniform float grounded;
uniform float landing;
uniform float sliding;
uniform vec4 terrain_a;
uniform vec4 terrain_b;
uniform vec3 terrain_c;
uniform int object_count;
uniform vec4 objects[24];
uniform sampler2D wave_tex;
in vec2 uv;
out vec4 fragColor;
const float PI=3.141592653589793;
float sat(float x){if(x<0.0)return 0.0;if(x>1.0)return 1.0;return x;}
float band(float x,float w){if(x>-w && x<w)return 1.0;return 0.0;}
vec2 rot(vec2 p,float a){return vec2(cos(a)*p.x-sin(a)*p.y,sin(a)*p.x+cos(a)*p.y);}
float pill(vec2 p,vec2 scale,float c){
 if(p.x < -scale.x*0.5 || p.x > scale.x*0.5 || p.y < -scale.y*0.5 || p.y > scale.y*0.5)return 0.0;
 return step(c,cos(PI*p.x/scale.x)*cos(PI*p.y/scale.y));
}
float waveHeight(float s){
 float p=s/level_length;
 p=sat(p);
 float a0=terrain_a.x;
 float gain=terrain_a.y;
 float k0=terrain_a.z;
 float k1=terrain_a.w;
 float amp=a0+gain*p;
 float chirp=(k1-k0)/level_length;
 float ph=terrain_b.x+k0*s+0.5*chirp*s*s;
 float shape=sin(ph)+terrain_b.y*sin(terrain_b.z*ph+terrain_b.w);
 shape+=terrain_c.x*sin(terrain_c.y*ph+terrain_c.z);
 return amp*shape;
}
void main(){
 float aspect=resolution.x/resolution.y;
 float wx=travel+(uv.x-0.30)*42.0;
 float gy=0.39+(waveHeight(wx)-camera_y)*0.075;
 vec3 sky=vec3(0.94,0.90,0.82)+vec3(0.025,0.018,0.006)*sin(PI*uv.y);
 vec3 col=sky;
 vec2 sunp=uv-vec2(0.72,0.76);
 float sun=step(sunp.x*sunp.x+sunp.y*sunp.y,0.0065);
 vec3 sunColor=vec3(0.84,0.76,0.64);
 col=mix(col,sunColor,sun*0.72);
 // Parallax ridges are sums of incommensurate sine modes.
 for(int mi=0;mi<4;mi++){
  float fm=float(mi);
  float ridge=0.43+fm*0.055+0.055*sin(5.2*uv.x+travel*(0.006+fm*0.002)+fm*1.7);
  ridge+=0.022*sin(11.3*uv.x-travel*0.004+fm);
  vec3 ridgeColor=vec3(0.74+fm*0.025,0.67+fm*0.035,0.57+fm*0.045);
  if(uv.y<ridge)col=mix(col,ridgeColor,0.34);
 }
 // Fine airborne wave diagrams echo the terrain equation.
 float cloudWave=0.72+0.022*sin(32.0*uv.x+time*0.08)+0.009*sin(73.6*uv.x+1.1);
 col=mix(col,vec3(0.60,0.50,0.40),band(uv.y-cloudWave,0.0012)*0.35);
 float field=texture(wave_tex,vec2(uv.x,sat((gy-uv.y)*2.0))).r;
 vec3 earth=vec3(0.105,0.092,0.078)+vec3(0.012,0.009,0.006)*sin(20.0*uv.y+field*3.0);
 if(uv.y<gy)col=earth;
 float crest=band(uv.y-gy,0.007);
 col=mix(col,vec3(0.985,0.965,0.915),crest);
 float contour=band(uv.y-(gy-0.035),0.0015);
 col=mix(col,vec3(0.57,0.47,0.36),contour*step(uv.y,gy));
 float under=band(uv.y-(gy-0.115-0.018*sin(wx*0.33)),0.0014);
 col=mix(col,vec3(0.76,0.61,0.43),under*step(uv.y,gy));
 // Pickups and hazards are anchored to the same analytic surface.
 for(int oi=0;oi<24;oi++){
  if(oi<object_count){
   vec4 ob=objects[oi];
   vec2 d=(uv-ob.xy)*vec2(aspect,1.0);
   float r2=d.x*d.x+d.y*d.y;
   float ring=step(r2,0.00072)-step(r2,0.00030);
   float halo=step(r2,0.00115)*0.18;
   float coinWave=band(d.y-0.005*sin(125.0*d.x),0.0017)*step(r2,0.00028);
   float spring=band(d.x-0.014*sin(190.0*d.y),0.0048)*step(-0.055,d.y)*step(d.y,0.055);
   float breaker=step(r2,0.00130+0.00042*sin(120.0*d.x)*sin(120.0*d.y));
   float breakerInner=step(r2,0.00083+0.00025*sin(120.0*d.x)*sin(120.0*d.y));
   float core=step(r2,0.00042);
   if(ob.z<0.5){col=mix(col,vec3(0.96,0.66,0.18),halo);col=mix(col,vec3(0.92,0.55,0.10),ring);col=mix(col,vec3(0.24,0.16,0.08),coinWave);}
   else if(ob.z<1.5){col=mix(col,vec3(0.10,0.48,0.78),step(r2,0.00105));col=mix(col,vec3(0.96,0.94,0.78),core);}
   else if(ob.z<2.5){col=mix(col,vec3(0.12,0.82,0.57),spring);col=mix(col,vec3(0.08,0.30,0.22),band(d.y+0.057,0.005)*step(r2,0.0022));}
   else {col=mix(col,vec3(0.98,0.90,0.78),breaker);col=mix(col,vec3(0.72,0.10,0.08),breakerInner);}
  }
 }
 // Surfer: procedural sine poses respond to slope, air speed, landing and input.
 vec2 surferPos=vec2(0.30,0.39+(surfer_y-camera_y)*0.075);
 vec2 q=(uv-surferPos)*vec2(aspect,1.0);
 float tumble=hurt*(0.45+0.35*sin(time*9.0));
 q=rot(q,-surfer_pitch-tumble);
 float air=1.0-grounded;
 float cadence=time*(4.0+0.32*speed);
 float ride=grounded*sin(cadence);
 float crouch=0.018*sliding+0.020*landing+0.010*air;
 float bob=0.004*ride*(1.0-sliding)-0.012*landing;
 float velocityLean=0.16*sin(0.055*surfer_vy);
 float bodyLean=-0.20-0.22*boost+0.35*air+velocityLean;
 float boardFlex=0.004*sin(cadence+PI*0.5)*grounded+0.007*landing;
 float board=pill(q-vec2(0.0,-0.022+boardFlex),vec2(0.165,0.019),0.12);
 float boardEdge=pill(q-vec2(0.0,-0.018+boardFlex),vec2(0.174,0.008),0.10);
 float body=pill(rot(q-vec2(-0.004,0.061-crouch+bob),bodyLean),vec2(0.042,0.086-0.018*sliding),0.08);
 vec2 headq=q-vec2(-0.020+0.012*air,0.119-crouch+bob);
 float head=step(headq.x*headq.x+headq.y*headq.y,0.00042);
 float legSwing=0.16*ride+0.22*air*sin(0.12*surfer_vy);
 float rearLeg=pill(rot(q-vec2(-0.027,0.019-crouch*0.45),-0.58+legSwing-0.30*sliding),vec2(0.020,0.074-0.020*landing),0.10);
 float frontLeg=pill(rot(q-vec2(0.026,0.020-crouch*0.45),0.58-legSwing+0.24*sliding),vec2(0.020,0.076-0.020*landing),0.10);
 float armWave=0.12*sin(cadence*0.62)+0.24*air-0.20*boost;
 float rearArm=pill(rot(q-vec2(-0.032,0.070-crouch),-0.72-armWave),vec2(0.016,0.063),0.10);
 float frontArm=pill(rot(q-vec2(0.025,0.071-crouch),0.82+armWave),vec2(0.016,0.066),0.10);
 float scarf=pill(rot(q-vec2(-0.060,0.091+0.008*sin(time*8.0)),-1.28),vec2(0.014,0.085),0.10);
 float limbs=sat(head+rearLeg+frontLeg+rearArm+frontArm);
 col=mix(col,vec3(0.055,0.050,0.046),limbs);
 col=mix(col,vec3(0.12,0.30,0.33),body);
 col=mix(col,vec3(0.86,0.55,0.20),sat(board+boardEdge));
 col=mix(col,vec3(0.72,0.16,0.10),scarf);
 if(boost>0.5){
  for(int ti=0;ti<6;ti++){
   float ft=float(ti);
   vec2 td=q-vec2(-0.09-ft*0.025,-0.018+0.008*sin(time*12.0+ft));
   float trail=step(td.x*td.x+td.y*td.y,0.00010);
   col=mix(col,vec3(0.75,0.48,0.22),trail*(1.0-ft/7.0)*(0.45+0.55*fuel/100.0));
  }
 }
 float vig=sin(PI*uv.x)*sin(PI*uv.y);
 col*=0.82+0.18*vig;
 fragColor=vec4(col,1.0);
}
"""



class PerspectiveLookup:
    """Monotone sampled quarter-sine used by CPU-projected world objects."""
    def __init__(self, samples: int = 2049):
        self.input = np.linspace(0.0, 1.0, samples, dtype=np.float64)
        self.depth = np.sin(0.5 * PI * self.input)

    def project(self, fraction: float) -> float:
        return float(np.interp(clamp01(fraction), self.input, self.depth))


class Renderer:
    """Wave Surfer renderer. The legacy runner shader is never selected here."""
    def __init__(self, width: int, height: int, standalone: bool):
        import moderngl
        self.width, self.height = width, height
        self.perspective = PerspectiveLookup()
        if standalone:
            try:
                self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
            except Exception:
                self.ctx = moderngl.create_standalone_context(require=330)
        else:
            self.ctx = moderngl.create_context(require=330)
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=SURF_FRAGMENT_SHADER)
        vertices = np.array((-1, -1, 1, -1, -1, 1, 1, 1), dtype="f4")
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "in_pos")
        self.color = self.ctx.texture((width, height), 4)
        self.wave = self.ctx.texture((128, 128), 1, dtype="f4")
        self.wave.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer((self.color,))
        self.program["wave_tex"].value = 0
        self.wave_version = -1
        self.present_program = None
        self.overlay = None
        if not standalone:
            self.present_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader="""#version 330
uniform sampler2D scene_tex;
uniform sampler2D overlay_tex;
in vec2 uv;
out vec4 fragColor;
void main(){
 vec3 scene=texture(scene_tex,uv).rgb;
 vec4 over=texture(overlay_tex,uv);
 fragColor=vec4(mix(scene,over.rgb,over.a),1.0);
}
""")
            self.present_vao = self.ctx.simple_vertex_array(self.present_program, self.vbo, "in_pos")
            self.overlay = self.ctx.texture((1280, 720), 4)
            self.overlay.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.present_program["scene_tex"].value = 0
            self.present_program["overlay_tex"].value = 1
        print(f"GL renderer: {self.ctx.info.get('GL_RENDERER', 'unknown')}")

    def render(self, state: SurferState, field: WaveField, alpha: float = 1.0, visual_time: float | None = None,
               show_objects: bool = True, readback: bool = True) -> bytes | None:
        render_distance = state.distance - state.speed * DT * (1.0 - clamp01(alpha))
        if render_distance < 0.0:
            render_distance = 0.0
        render_y = state.y - state.vertical_velocity * DT * (1.0 - clamp01(alpha))
        if field.version != self.wave_version:
            self.wave.write(field.u.tobytes())
            self.wave_version = field.version
        self.wave.use(0)
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.program["resolution"].value = (float(self.width), float(self.height))
        self.program["time"].value = state.time if visual_time is None else visual_time
        self.program["travel"].value = render_distance
        self.program["level_length"].value = state.level.length
        self.program["surfer_y"].value = render_y
        self.program["camera_y"].value = state.camera_y
        self.program["surfer_pitch"].value = state.pitch
        self.program["surfer_vy"].value = state.vertical_velocity
        self.program["speed"].value = state.speed
        self.program["fuel"].value = state.fuel
        self.program["boost"].value = 1.0 if state.boosting else 0.0
        self.program["hurt"].value = 1.0 if state.invulnerable > 0.0 else 0.0
        self.program["grounded"].value = 1.0 if state.grounded else 0.0
        self.program["landing"].value = state.landing_compression
        self.program["sliding"].value = 1.0 if state.sliding else 0.0
        params = terrain_parameters(state.level.seed, state.level.domain.stage, state.level.domain.difficulty)
        self.program["terrain_a"].value = params[0:4]
        self.program["terrain_b"].value = params[4:8]
        self.program["terrain_c"].value = params[8:11]
        packed = np.zeros((24, 4), dtype="f4")
        kind_id = {"coin": 0.0, "boost": 1.0, "spring": 2.0, "breaker": 3.0}
        count = 0
        if show_objects:
            for idx, thing in enumerate(state.level.things):
                dx = thing.distance - render_distance
                if dx > 32.0:
                    break
                if idx in state.seen or dx < -2.0:
                    continue
                ground, _, _ = surf_geometry(state.level, thing.distance)
                screen_x = 0.30 + dx / 42.0
                screen_y = 0.39 + (ground - state.camera_y) * 0.075
                if thing.kind in ("coin", "boost"):
                    screen_y += 0.075 + 0.025 * math.sin(thing.phase)
                packed[count] = (screen_x, screen_y, kind_id[thing.kind], thing.phase)
                count += 1
                if count >= 24:
                    break
        self.program["object_count"].value = count
        self.program["objects"].write(packed.tobytes())
        self.vao.render(mode=5)
        if readback:
            return self.fbo.read(components=3, alignment=1)
        return None

    def present(self, overlay_surface: object, window_size: tuple[int, int]) -> None:
        """Composite entirely on the GPU, avoiding a framebuffer readback."""
        import pygame
        if self.present_program is None or self.overlay is None:
            raise RuntimeError("present() requires a window-backed renderer")
        self.overlay.write(pygame.image.tobytes(overlay_surface, "RGBA", True))
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, window_size[0], window_size[1])
        self.ctx.disable(self.ctx.BLEND)
        self.color.use(0)
        self.overlay.use(1)
        self.present_vao.render(mode=5)



