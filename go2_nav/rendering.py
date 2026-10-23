"""Mesh-only, streaming H.264 recording with an optional robot close-up."""
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from go2_nav.fonts import font as load_font


def visual_options():
    option=mujoco.MjvOption()
    option.geomgroup[:]=[1,0,1,1,1,0]
    return option

class Recorder:
    def __init__(self,model,path,width=1280,height=720,fps=25,complex_scene=False,inset=True):
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
        model.vis.global_.offheight=max(model.vis.global_.offheight,height)
        self.model=model;self.width=width;self.height=height;self.fps=fps;self.complex=complex_scene;self.options=visual_options()
        self.renderer=mujoco.Renderer(model,height,width)
        self.close=mujoco.Renderer(model,240,400) if inset else None
        self.writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
        self.font=load_font(20)
        self.small=load_font(16)
        self.frames=0;self.follow=False;self.after_draw=None;self.details=None
    def append(self,data,position,goal,command,title,observation=None,visibility=None):
        camera=mujoco.MjvCamera();camera.lookat[:]=[6 if self.complex else 3.5,0,.4];camera.distance=17 if self.complex else 10;camera.azimuth=90;camera.elevation=-62
        if self.follow:
            camera.lookat[:]=position+np.array([.3,0,.25]);camera.distance=7;camera.azimuth=110;camera.elevation=-18
        self.renderer.update_scene(data,camera,scene_option=self.options);frame=Image.fromarray(self.renderer.render())
        draw=ImageDraw.Draw(frame);draw.rectangle((0,0,self.width,96 if self.details else 70),fill=(20,27,36));draw.text((20,10),'NavRL + Go2 | '+title,fill=(245,248,255),font=self.font)
        draw.text((20,40),f'Time {data.time:5.2f}s    Goal distance {np.linalg.norm(position[:2]-goal):4.2f}m    Command vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f}',fill=(180,205,225),font=self.small)
        if self.details:draw.text((20,68),self.details(data),fill=(255,220,125),font=self.small)
        draw.rectangle((0,self.height-28,self.width,self.height),fill=(20,27,36));draw.text((16,self.height-24),'Green: goal   Blue: moving obstacles   Mesh visuals only / collision shapes hidden',fill=(210,220,230),font=self.small)
        if self.close:
            follow=mujoco.MjvCamera();follow.lookat[:]=position+np.array([.1,0,.05]);follow.distance=2.4;follow.azimuth=125;follow.elevation=-25
            self.close.update_scene(data,follow,scene_option=self.options);picture=Image.fromarray(self.close.render())
            x,y=self.width-418,self.height-290;draw.rounded_rectangle((x-3,y-27,x+403,y+243),radius=8,fill=(20,27,36),outline=(105,150,185),width=2)
            frame.paste(picture,(x,y));draw.text((x+10,y-23),'Go2 visual mesh / close-up',fill=(235,245,255),font=self.small)
        if observation is not None:
            from go2_nav.common.input_visualization import radar_panel
            panel=radar_panel(observation,visibility=visibility);frame.paste(panel,(14,self.height-panel.height-36))
        if self.after_draw:self.after_draw(frame,draw,data)
        self.writer.append_data(np.asarray(frame));self.frames+=1
    def close_writer(self):
        self.writer.close();self.renderer.close()
        if self.close:self.close.close()
