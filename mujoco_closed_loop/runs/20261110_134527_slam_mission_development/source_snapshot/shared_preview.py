"""Bounded RGB-D preview transport; the control process never reads image bytes."""
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

class SharedPreview:
    def __init__(self,spec=None,lock=None):
        self.owner=spec is None;self.closed=False
        self.lock=lock if lock is not None else mp.get_context('spawn').Lock()
        height,width=360,480;header_size=5*8;rgb_size=height*width*3;depth_size=height*width*4
        self.memory=shared_memory.SharedMemory(create=self.owner,size=header_size+2*rgb_size+depth_size if self.owner else 0,name=None if self.owner else spec['name'])
        self.spec=dict(name=self.memory.name,height=height,width=width)
        self.header=np.ndarray((5,),np.float64,buffer=self.memory.buf,offset=0)
        self.nav_rgb=np.ndarray((height,width,3),np.uint8,buffer=self.memory.buf,offset=header_size)
        self.wrist_rgb=np.ndarray((height,width,3),np.uint8,buffer=self.memory.buf,offset=header_size+rgb_size)
        self.depth=np.ndarray((height,width),np.float32,buffer=self.memory.buf,offset=header_size+2*rgb_size)
        if self.owner:self.header[:]=[-1,0,0,float('inf'),float('inf')]

    def write(self,stamp,nav,wrist,nav_valid,wrist_valid,nav_error,wrist_error):
        if not self.lock.acquire(False):return False
        try:
            np.copyto(self.nav_rgb,nav.rgb);np.copyto(self.wrist_rgb,wrist.rgb);np.copyto(self.depth,wrist.depth_m)
            self.header[:]=[stamp,nav_valid,wrist_valid,nav_error,wrist_error]
            return True
        finally:self.lock.release()

    def read(self,maximum_stamp=float('inf')):
        if not self.lock.acquire(False):return None
        try:
            stamp=float(self.header[0])
            if stamp<0 or stamp>maximum_stamp:return None
            return dict(stamp=stamp,nav_rgb=self.nav_rgb.copy(),wrist_rgb=self.wrist_rgb.copy(),wrist_depth_m=self.depth.copy(),nav_valid=bool(self.header[1]),wrist_valid=bool(self.header[2]),nav_reprojection_px=float(self.header[3]),wrist_reprojection_px=float(self.header[4]))
        finally:self.lock.release()

    def close(self):
        if self.closed:return
        self.closed=True;self.header=None;self.nav_rgb=None;self.wrist_rgb=None;self.depth=None
        self.memory.close()
        if self.owner:
            try:self.memory.unlink()
            except FileNotFoundError:pass
