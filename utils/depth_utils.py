import torch
from torch.functional import norm
#from torch._C import double
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timeit
import torch
"""
Calculates surface normal maps given dense depth maps. Uses Simple Haar feature kernels to calculate surface gradients
"""

class SurfaceNet(nn.Module):

    def __init__(self):
        super(SurfaceNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.convDelYDelZ = nn.Conv2d(1, 1, 3)
        self.convDelXDelZ = nn.Conv2d(1, 1, 3)
        if torch.cuda.is_available():
             dev = "cuda:0"
        else:
             dev = "cpu" 
        self.device = torch.device(dev)
        # # print("dev!!!", dev)  

    def forward(self, x, fx, fy):
        #start = timeit.default_timer()
        #x = x.to(self.device)
        nb_channels = 1#x.shape[1]
        h, w = x.shape[-2:]        

        dz_dv, dz_du = torch.gradient(x, dim=[1,2])       
        du_dx = fx / x  # x is xyz of camera coordinate
        dv_dy = fy / x 

        dz_dx = dz_du * du_dx
        dz_dy = dz_dv * dv_dy

        dz_dz = torch.ones(dz_dy.shape, dtype=torch.float64).to(self.device)
        
        #print('kernel',delzdelx.shape)
        normal_cross = torch.stack((-dz_dx, -dz_dy, dz_dz), 0)
        
        surface_norm = torch.div(normal_cross,  norm(normal_cross, dim=0))
        # * normal vector space from [-1.00,1.00] to [0,255] for visualization processes
        #surface_norm_viz = torch.mul(torch.add(surface_norm, 1.00000),127 )
        
        #end = timeit.default_timer()
        #print("torch method time", end-start)
        return torch.stack((surface_norm[1,:,:], -surface_norm[1,:,:], torch.zeros(dz_dy.shape, dtype=torch.float64).to(self.device))).squeeze(1)

def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))

def get_surface_normal_by_depth(depth, fovx, fovy, K=None, ):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camera's intrinsic
    """
    #K = [[1.0, 0], [0, 1.0]] if K is None else K
    #fx, fy = K[0][0], K[1][1]

    depth = depth.squeeze(0)

    fx = fov2focal(fovx, depth.shape[0])
    fy = fov2focal(fovy, depth.shape[1])    

    dz_dv, dz_du = torch.gradient(depth, dim=[0,1])  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = torch.stack((-dz_dx, -dz_dy, torch.ones_like(depth)), dim=0)
    # normalize to unit vector
    #normal_unit = torch.nn.functional.normalize(normal_cross, dim=0)
    normal_unit = (normal_cross / torch.functional.norm(normal_cross, dim=0, keepdim=True))
    # set default normal to [0, 0, 1]
    #normal_unit[~torch.isfinite(normal_unit).all(2)] = torch.tensor([0, 0, 1])
    reshaped = normal_unit.reshape(3, -1)
    reshaped = reshaped.transpose(1,0)
    reshaped.nan_to_num_(-torch.inf)
    reshaped = torch.where(~torch.isfinite(reshaped), torch.tensor([0, 0, 0], device="cuda"), reshaped)
    reshaped = reshaped.transpose(1,0)

    normal_unit = reshaped.reshape(3, depth.shape[0], depth.shape[1])
    #reshaped[~torch.isfinite(reshaped)]  = torch.tensor([0, 0, 1])
    #normal_unit[~torch.any(normal_unit.isnan(),dim=1)] = torch.tensor([0, 0, 1])

    return normal_unit