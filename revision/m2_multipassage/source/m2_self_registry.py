"""Partition the original physical collision registry without changing the MuJoCo model."""
import numpy as np
from itertools import combinations

def partition(model, original_pairs):
    pairs=np.asarray(original_pairs,dtype=int).reshape(-1,2)
    mount=model.body('mount').id
    cylinders=[i for i in range(model.ngeom) if model.geom_bodyid[i]==mount
               and model.geom_type[i]==5 and not model.geom(i).name
               and np.allclose(model.geom_size[i,:2],[.0675,.0352],rtol=0,atol=1e-8)]
    if len(cylinders)!=1:
        raise ValueError('Central mount identification differs from the registered SUPER CAD assembly')
    fit={tuple(sorted((cylinders[0],model.geom('arm1').id)))}
    fingers=[int(model.geom('collision_gripper_finger'+str(i)).id) for i in range(3)]
    fit.update(tuple(sorted(pair)) for pair in combinations(fingers,2))
    is_fit=np.asarray([tuple(sorted(map(int,row))) in fit for row in pairs],dtype=bool)
    if is_fit.sum()!=len(fit):
        raise ValueError('The registered mount-to-proximal fit pair must be present exactly once')
    return pairs[~is_fit].copy(),pairs[is_fit].copy()
