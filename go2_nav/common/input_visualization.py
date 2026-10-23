"""Visible environment-derived observations, without feeding a global map."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from go2_nav.fonts import font as load_font


def radar_panel(observation,size=(340,260),visibility=None):
    state,lidar,dynamic=observation;state=np.asarray(state).reshape(-1);lidar=np.asarray(lidar).reshape(36,4);dynamic=np.asarray(dynamic).reshape(5,10)
    panel=Image.new('RGB',size,(15,25,36));draw=ImageDraw.Draw(panel);font=load_font(14);cx,cy=size[0]/2,size[1]/2+10;scale=23
    draw.text((10,8),'Policy input: body-frame local sensing',font=font,fill='white')
    for radius in (1,2,3,4):draw.ellipse((cx-radius*scale,cy-radius*scale,cx+radius*scale,cy+radius*scale),outline=(45,65,80))
    for i,distance in enumerate(4-lidar[:,1]):
        angle=i*np.pi/18
        if visibility is not None and visibility[i]<.99:
            ex=cx-np.sin(angle)*4*scale;ey=cy-np.cos(angle)*4*scale
            draw.line((cx,cy,ex,ey),fill=(115,115,130),width=2)
            continue
        end=(cx-np.sin(angle)*distance*scale,cy-np.cos(angle)*distance*scale);color=(80,190,125) if distance>1.5 else (235,90,65);draw.line((cx,cy,*end),fill=color,width=1)
    gx,gy=state[:2];draw.line((cx,cy,cx-gy*90,cy-gx*90),fill=(50,245,80),width=3)
    draw.polygon([(cx,cy-10),(cx-6,cy+7),(cx+6,cy+7)],fill='white')
    for item in dynamic:
        if item[3]<=0:continue
        xy=item[:2]*item[3];point=np.array([cx-xy[1]*scale,cy-xy[0]*scale]);velocity=item[5:7];end=point+[-velocity[1]*scale,-velocity[0]*scale]
        draw.ellipse((* (point-4),*(point+4)),fill=(80,150,255));draw.line((*point,*end),fill=(100,175,255),width=3)
    draw.text((10,size[1]-23),f'Range 4m | Goal {state[3]:.1f}m | blue: moving',font=font,fill=(195,215,230));return panel


def save_observation_figure(env,index,path):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon
    observation=tuple(o[index] for o in env.observation);state,lidar,dynamic=observation;info=env.infos[index];data=env.data[index];model=env.model
    fig,axes=plt.subplots(2,2,figsize=(14,10));ax=axes[0,0]
    for gid in np.flatnonzero(model.geom_group==3):
        c=data.geom_xpos[gid,:2];sz=model.geom_size[gid];kind=model.geom_type[gid]
        if kind==6:
            R=data.geom_xmat[gid].reshape(3,3)[:2,:2];p=np.array([[-sz[0],-sz[1]],[sz[0],-sz[1]],[sz[0],sz[1]],[-sz[0],sz[1]]])@R.T+c;ax.add_patch(Polygon(p,color=model.geom_rgba[gid,:3],alpha=.5))
        else:ax.add_patch(Circle(c,sz[0],color=model.geom_rgba[gid,:3],alpha=.5))
    pos=info['position'][:2];ax.plot(*pos,'ko',label='Robot');ax.plot(*env.goal[index],'g*',ms=15,label='Goal');ax.add_patch(Circle(pos,4,fill=False,ls='--',color='blue',label='Local 4m range'));ax.set(xlim=(-1,14),ylim=(-5,5),title='Simulation truth (display only, NOT a policy map input)');ax.set_aspect('equal');ax.legend(fontsize=8,loc='upper left')
    ax=axes[0,1];ranges=4-lidar[0,:,1];theta=np.arange(36)*np.pi/18
    for angle,d in zip(theta,ranges):ax.plot([0,np.cos(angle)*d],[0,np.sin(angle)*d],color='green' if d>1.5 else 'red',alpha=.6)
    ax.arrow(0,0,state[0]*3,state[1]*3,color='lime',head_width=.12);
    for item in dynamic[0]:
        if item[3]<=0:continue
        point=item[:2]*item[3];ax.plot(*point,'bo');ax.arrow(*point,*item[5:7],color='blue',head_width=.1)
    ax.set(xlim=(-4.3,4.3),ylim=(-4.3,4.3),xlabel='body x / m',ylabel='body y / m',title='36 local rays + relative goal direction');ax.set_aspect('equal');ax.grid(alpha=.2)
    image=axes[1,0].imshow(lidar[0].T,aspect='auto',origin='lower',vmin=0,vmax=3.9,cmap='magma');axes[1,0].set(xlabel='horizontal ray index (10 degrees)',ylabel='vertical feature layer',title='Exact actor lidar tensor: 4 - distance (1 x 36 x 4)');fig.colorbar(image,ax=axes[1,0],label='encoded closeness')
    image=axes[1,1].imshow(dynamic[0],aspect='auto',vmin=-2,vmax=4,cmap='coolwarm');axes[1,1].set(xlabel='feature index',ylabel='nearest obstacle slot',title='Exact actor dynamic tensor (1 x 5 x 10)');axes[1,1].set_xticks(range(10),['nx','ny','nz','dxy','dz','vx','vy','vz','width','height']);fig.colorbar(image,ax=axes[1,1]);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
