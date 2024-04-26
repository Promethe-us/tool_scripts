import numpy as np
from plyfile import PlyData

def load_ply(ply_path):
    max_sh_degree = 3
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    C0 = 0.28209479177387814
    def RGB2SH(rgb): return (rgb - 0.5) / C0
    def SH2RGB(sh): return sh * C0 + 0.5
        
    fused_color = features_dc[..., 0] # (N, 3)
    rgb = SH2RGB(fused_color) # (N, 3)
    
    return xyz, rgb
