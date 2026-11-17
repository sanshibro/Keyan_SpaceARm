import numpy as np
from estimation import relative_camera,orientation_shape
# Nonzero UAV translation and rotation must cancel from recovered arm shape.
a=.7;c,s=np.cos(a),np.sin(a)
U=np.eye(4);U[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];U[:3,3]=[3,2,1]
B=np.diag([-1.,1.,-1.,1.]);B[:3,3]=[0,0,-.03]
T=np.eye(4);a=.5;c,s=np.cos(a),np.sin(a);T[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];T[:3,3]=[.05,0,.3]
C=np.eye(4);C[:3,3]=[.061,0,.065]
assert np.allclose(relative_camera(U,B,U@B@T@C,1.,1.),T@C)
assert np.allclose(orientation_shape(relative_camera(U,B,U@B@T@C,1.,1.),C),[.5,0])
try:relative_camera(U,B,U@B@T@C,1.,1.1)
except ValueError:pass
else:raise AssertionError('Stale pose accepted')
print('PASS: UAV motion compensation, known bend recovery, stale timestamp rejection')
