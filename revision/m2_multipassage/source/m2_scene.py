"""M2 shared industrial corridor geometry, in metres (development registration).

All visible structural solids have an identical COL_* physical primitive.
Natural surface textures are shared by the wrist and observer cameras.
No trajectory, simulator truth, or route is supplied by this module.
"""
import numpy as np

AXIS=np.array([.6109852853641633,.4616567134559925,-.6430941299603677])
FACE=np.array([4.95,.18,1.65]); STANDOFF=.168
HOVER=np.array([0.,-.10,2.12])
LENGTH=5.2
GATES=(dict(name='gate1',x=1.15,y=-.10,z=2.13,width=.86,height=.74),
       dict(name='gate2',x=2.55,y=.18,z=1.99,width=None,height=None),
       dict(name='gate3',x=3.95,y=-.14,z=2.10,width=.86,height=.74))

def boxes():
    out=[]
    def add(n,lo,hi,material):out.append((n,np.asarray(lo,float),np.asarray(hi,float),material))
    # Native launch is at x=0,z=1.5: the lowered entry platform leaves the
    # straight arm and landing gear free throughout the vertical departure.
    add('entry_deck',[-.55,-.82,.41],[.60,.82,.55],'deck')
    add('deck',[.60,-.82,.41],[5.45,.82,1.25],'deck')
    add('far_wall',[-.55,.80,1.25],[5.45,.84,2.82],'wall')
    # A permanently transparent inspection guard is identical in every camera.
    add('near_guard',[-.55,-.84,1.25],[5.45,-.80,2.82],'glass')
    add('roof_edge_far',[-.55,.72,2.76],[5.45,.80,2.82],'steel')
    add('roof_edge_near',[-.55,-.80,2.76],[5.45,-.72,2.82],'steel')
    for g in (GATES[0],GATES[2]):
        x,y,z,w,h=g['x'],g['y'],g['z'],g['width'],g['height'];n=g['name']
        add(n+'_sill',[x-.055,-.80,1.25],[x+.055,.80,z-h/2],'bulkhead')
        add(n+'_lintel',[x-.055,-.80,z+h/2],[x+.055,.80,2.82],'bulkhead')
        add(n+'_negative',[x-.055,-.80,z-h/2],[x+.055,y-w/2,z+h/2],'bulkhead')
        add(n+'_positive',[x-.055,y+w/2,z-h/2],[x+.055,.80,z+h/2],'bulkhead')
    add('overhead_duct',[2.18,-.80,2.23],[2.94,.80,2.68],'duct')
    add('low_service_plinth',[2.13,-.80,1.25],[2.94,.80,1.39],'steel')
    add('terminal_panel',[5.25,-.80,1.25],[5.30,.80,2.82],'wall')
    return out

def pipes():
    return [
      ('offset_riser_negative',np.array([2.44,-.36,1.39]),np.array([2.44,-.36,2.23]),.045,'copper'),
      ('offset_riser_positive',np.array([2.67,.70,1.39]),np.array([2.67,.70,2.23]),.045,'copper'),
      ('branch_negative',np.array([2.44,-.36,1.66]),np.array([2.83,-.36,1.66]),.026,'steel'),
      ('branch_cross',np.array([2.83,-.36,1.66]),np.array([2.83,.25,1.66]),.026,'steel'),
      ('far_header',np.array([.05,.72,2.64]),np.array([5.08,.72,2.64]),.036,'copper'),
      ('inspection_flange',FACE,FACE+.223*AXIS,.1075,'flange'),
      ('flange_support',FACE+.20*AXIS,np.array([5.24,.53,1.38]),.040,'steel')]

CAMERAS=(
 ('observer_oblique',[6.65,-6.9,4.75],[2.55,0,1.97],32),
 ('observer_side',[2.55,-8.0,2.32],[2.55,0,1.99],27),
 ('observer_dock',[4.55,-.72,2.32],[4.95,.18,1.65],46),
 ('observer_end',[5.98,-.03,2.25],[1.95,.02,2.03],52))
