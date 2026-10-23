"""Mixed-shape local navigation courses; no map is passed to the RL actor."""
import math
import xml.etree.ElementTree as ET

COMPLEX_SCENARIOS=('warehouse','warehouse_dynamic','dense','dense_dynamic')

def add_obstacle(world,name,kind,pos,size,yaw=0,color='.76 .34 .18 1'):
    return ET.SubElement(world,'geom',name=name,type=kind,pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),euler=f'0 0 {yaw}',group='3',contype='2',conaffinity='1',rgba=color,friction='1 .01 .001')

def motions(scenario):
    if scenario=='dynamic':return [('moving_obstacle',(4,0,.6),1.5,.5,0,1,.35)]
    if scenario in ('warehouse_dynamic','dense_dynamic'):
        return [('moving_obstacle_0',(5.2,0,.65),1.5,.32,0,1,.32),('moving_obstacle_1',(10.1,0,.65),1.7,.27,1.6,1,.36)]
    return []

def populate(world,scenario):
    if scenario in ('forest','forest_sprint'):
        from .forest_scene import populate_forest
        populate_forest(world,approach=scenario=='forest_sprint')
    if scenario in ('static','dynamic'):
        for i,(x,y,r) in enumerate([(2.5,0.,.45),(4.5,1.6,.4),(5.,-1.6,.5)]):
            add_obstacle(world,f'obstacle_{i}','cylinder',(x,y,.6),(r,.6))
    if scenario in COMPLEX_SCENARIOS:
        # A wide industrial aisle with staggered cargo, columns and barriers.
        add_obstacle(world,'obstacle_north_wall','box',(6,4.5,.7),(7,.12,.7),color='.35 .42 .5 1')
        add_obstacle(world,'obstacle_south_wall','box',(6,-4.5,.7),(7,.12,.7),color='.35 .42 .5 1')
        boxes=[(2.7,.75,.45,.65,.22),(5.3,2.4,.8,.45,-.15),(7.5,-3.2,.7,.45,-.25),(10.6,2.8,.7,.5,.3),(1.1,3.3,.8,.5,0),(2.8,-3.6,.65,.4,.15),(8.8,3.5,1.,.45,0),(11.4,-3.4,.6,.5,-.1)]
        for i,(x,y,sx,sy,yaw) in enumerate(boxes):
            height=.55+.12*(i%3)
            add_obstacle(world,f'obstacle_crate_{i}','box',(x,y,height),(sx,sy,height),yaw,color=['.75 .46 .16 1','.45 .55 .64 1','.63 .37 .22 1'][i%3])
            # Noncolliding top slats add visible detail to the cargo.
            for j in (-1,1):
                offset=j*sx*.6
                ET.SubElement(world,'geom',name=f'crate_slat_{i}_{j}',type='box',pos=f'{x+offset*math.cos(yaw)} {y+offset*math.sin(yaw)} {2*height+.015}',size=f'.035 {sy*.95} .015',euler=f'0 0 {yaw}',contype='0',conaffinity='0',density='0',group='4',rgba='.24 .2 .14 1')
        for i,(x,y,r) in enumerate([(4.4,-2.8,.42),(9.3,.15,.5),(1.5,-3,.36),(6.7,-3.4,.4),(11.8,2,.35),(3.8,3.7,.3)]):
            add_obstacle(world,f'obstacle_column_{i}','cylinder',(x,y,.8),(r,.8),color='.78 .23 .15 1')
        add_obstacle(world,'obstacle_rounded_barrier','capsule',(6.6,.85,.55),(.27,.7),color='.75 .63 .12 1').set('quat','0.70710678 0.70710678 0 0')
        # quat and euler cannot both describe the same orientation.
        world.find("geom[@name='obstacle_rounded_barrier']").attrib.pop('euler')
    if scenario in ('dense','dense_dynamic'):
        # Extra cargo islands preserve a roughly 2 m central passage.
        for i,(x,y,sx,sy) in enumerate([(0.8,-2.2,.35,.35),(1.4,1.9,.35,.45),(4.3,2.3,.35,.35),(5.5,-2.8,.3,.3),(7.6,2.7,.4,.35),(8.8,-2.4,.3,.3),(10.2,-2.6,.4,.35),(12.3,-1.7,.3,.4)]):
            add_obstacle(world,f'obstacle_dense_crate_{i}','box',(x,y,.5),(sx,sy,.5),.1*(i%3-1),color='.34 .53 .47 1')
    for name,center,amplitude,frequency,phase,axis,radius in motions(scenario):
        b=ET.SubElement(world,'body',name=name,mocap='true',pos=' '.join(map(str,center)))
        ET.SubElement(b,'geom',name='obstacle_dynamic_'+name,type='cylinder',size=f'{radius} {center[2]}',group='3',contype='2',conaffinity='1',rgba='.15 .38 .88 1')
        ET.SubElement(b,'geom',name='dynamic_cap_'+name,type='cylinder',pos=f'0 0 {center[2]+.01}',size=f'{radius*.88} .015',density='0',group='4',contype='0',conaffinity='0',rgba='.55 .78 1 1')

def update_motions(model,data,scenario):
    velocities={}
    for name,center,amplitude,frequency,phase,axis,radius in motions(scenario):
        mid=model.body(name).mocapid[0];position=list(center);position[axis]+=amplitude*math.sin(frequency*data.time+phase);data.mocap_pos[mid]=position
        velocity=[0.,0.];velocity[axis]=amplitude*frequency*math.cos(frequency*data.time+phase)
        velocities[model.geom('obstacle_dynamic_'+name).id]=velocity
    return velocities
