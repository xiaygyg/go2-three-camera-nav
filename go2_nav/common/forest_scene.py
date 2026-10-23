"""Deterministic forest with traversable patrol corridors; never an actor input."""
import xml.etree.ElementTree as ET
import numpy as np

BOUNDS=(-6.,60.,-23.,23.)
DEFAULT_GOALS=np.array([[18.,0.],[36.,13.],[53.,0.],[35.,-14.],[12.,-9.]])
TREE_COUNT=260
SEED=20261014

def segment_distance(p,a,b):
    line=b-a
    return np.linalg.norm(p-(a+np.clip((p-a)@line/max(line@line,1e-9),0,1)*line))

def layout(approach=False):
    rng=np.random.default_rng(SEED)
    route=np.vstack([[-46.,0.],[0.,0.],DEFAULT_GOALS]) if approach else np.vstack([[0.,0.],DEFAULT_GOALS])
    trees=[]
    for _ in range(100000):
        if len(trees)==TREE_COUNT:break
        p=rng.uniform([BOUNDS[0]+1,BOUNDS[2]+1],[BOUNDS[1]-1,BOUNDS[3]-1])
        r=rng.uniform(.18,.38)
        # A 3.4 m wide patrol corridor leaves room for the entire quadruped.
        if min(segment_distance(p,a,b) for a,b in zip(route[:-1],route[1:]))<1.7+r:continue
        if any(np.linalg.norm(p-q[:2])<r+q[2]+1.35 for q in trees):continue
        trees.append([*p,r,rng.uniform(4.8,7.0)])
    if len(trees)!=TREE_COUNT:raise RuntimeError('Forest packing failed')
    rocks=[]
    for _ in range(3000):
        if len(rocks)==30:break
        p=rng.uniform([-4,-21],[58,21]);r=rng.uniform(.28,.55)
        if min(segment_distance(p,a,b) for a,b in zip(route[:-1],route[1:]))<1.7+r:continue
        if any(np.linalg.norm(p-q[:2])<r+q[2]+.9 for q in trees+rocks):continue
        rocks.append([*p,r,rng.uniform(.3,.6)])
    if approach:
        trees.extend([[6.,0.,.26,5.8],[12.,.7,.26,5.4],[26.,6.,.24,6.],[44.,7.,.30,5.9]])
    return np.asarray(trees),np.asarray(rocks)

def populate_forest(world,approach=False):
    trees,rocks=layout(approach)
    if approach:
        ET.SubElement(world,'geom',name='approach_road',type='box',pos='-26 0 .006',size='20 1.3 .006',group='4',contype='0',conaffinity='0',density='0',rgba='.27 .28 .23 1')
        for x in range(-45,-6,3):
            for y in (-1.2,1.2):
                ET.SubElement(world,'geom',type='box',pos=f'{x} {y} .02',size='.7 .025 .005',group='4',contype='0',conaffinity='0',density='0',rgba='.8 .83 .7 1')
        for y in (-2,2):
            ET.SubElement(world,'geom',type='cylinder',pos=f'-6 {y} 1.1',size='.08 1.1',group='4',contype='0',conaffinity='0',density='0',rgba='.1 .7 .8 1')
    for i,(x,y,r,h) in enumerate(trees):
        ET.SubElement(world,'geom',name=f'obstacle_tree_{i}',type='cylinder',pos=f'{x} {y} {h/2}',size=f'{r} {h/2}',group='3',contype='2',conaffinity='1',rgba='.29 .17 .08 1',friction='1 .01 .001')
        for j in range(3):
            # Layered, visual-only crowns retain a clear ground-level sightline.
            z=h-.6+j*.65
            ET.SubElement(world,'geom',name=f'forest_crown_{i}_{j}',type='ellipsoid',pos=f'{x} {y} {z}',size=f'{1.05-j*.19} {1.05-j*.19} .7',group='4',contype='0',conaffinity='0',density='0',rgba=['.13 .31 .10 1','.19 .39 .12 1','.26 .46 .17 1'][(i+j)%3])
    for i,(x,y,r,h) in enumerate(rocks):
        ET.SubElement(world,'geom',name=f'obstacle_rock_{i}',type='cylinder',pos=f'{x} {y} {h/2}',size=f'{r} {h/2}',group='3',contype='2',conaffinity='1',rgba='.4 .43 .34 1',friction='1 .01 .001')
        ET.SubElement(world,'geom',name=f'forest_rock_cap_{i}',type='ellipsoid',pos=f'{x} {y} {h*.7}',size=f'{r} {r*.9} {h*.5}',group='4',contype='0',conaffinity='0',density='0',rgba='.46 .49 .39 1')
    for i,(x,y) in enumerate(DEFAULT_GOALS):
        ET.SubElement(world,'geom',name=f'patrol_waypoint_{i}',type='cylinder',pos=f'{x} {y} .025',size='.4 .025',group='4',contype='0',conaffinity='0',density='0',rgba='.2 .55 .8 .8')
