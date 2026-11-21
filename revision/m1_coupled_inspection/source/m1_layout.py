"""Surveyed industrial gallery, in ENU metres. Every camera uses this one scene."""
import numpy as np

AXIS=np.array([0.6109852853641633,0.4616567134559925,-0.6430941299603677])
FACE=np.array([3.54,.28,1.775])
STANDOFF=.168
GATE_X=1.
SILL=1.695
LINTEL=2.365
GATE_HALF=.395
HOVER=np.array([0.,0.,2.15])
DEPLOY=np.array([-.02,0.,2.15])
FROZEN='881a7996739479874b489ec910d7c78ebb26c075'
METHODS=('straight','fixed','sequential','unregularized','coupled')
SEEDS=(26112161,26112167,26112173)
LABELS={'straight':'Straight arm','fixed':'Fixed morphology','sequential':'AET-Seq. adaptation',
        'unregularized':'Joint, no coupling cost','coupled':'Coupled whole-body'}
# The portal closes the entire flight corridor at x = 1.00. The opposing
# service boxes share the same geometry in the map, physics and RGB-D scene.
def blocks():
    return {
       'portal_sill':([.95,-.82,.0],[1.05,.82,SILL]),
       'portal_lintel':([.95,-.82,LINTEL],[1.05,.82,2.66]),
       'portal_left':([.95,-.82,SILL],[1.05,-GATE_HALF,LINTEL]),
       'portal_right':([.95,GATE_HALF,SILL],[1.05,.82,LINTEL]),
       'approach_deck':([-.55,-.82,0.],[.585,.82,.825]),
       'gallery_deck':([.585,-.82,0.],[4.14,.82,1.265]),
       'backwall':([-.54,.745,.80],[4.14,.78,2.58]),
       'near_guard':([-.12,-.65,1.28],[3.20,-.615,2.49]),
       'roof':([-.09,-.65,2.52],[3.94,.77,2.555]),
       'service_spine':([1.96,-.615,1.31],[2.15,-.22,1.745]),
       'exit_baffle':([2.44,.29,1.32],[2.59,.745,1.845]),
       'return_cabinet':([2.92,-.615,1.27],[3.07,-.245,2.08])}
def pipes():
    return {'asymmetric_riser':([1.77,.135,1.265],[1.77,.135,2.515],.047),
            'riser_return':([1.77,.137,2.463],[1.77,.735,2.463],.044),
            'instrument_tap':([1.650,.135,1.825],[1.770,.135,1.825],.019),
            'instrument_elbow':([1.650,-.115,1.825],[1.650,.135,1.825],.019)}
def longitudinal(x):
    return np.interp(x,[-1.2,.58,1.07,2.55,3.0],[-1.2,.58,1.07,4.11,4.56])
