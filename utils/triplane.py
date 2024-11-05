import torch
import torch.nn as nn
import nvdiffrast.torch

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# def get_unique_filename(base_filename, extension):
#     """
#     Generate a unique file name for saving figures.
    
#     Args:
#     - base_filename: the base name of the file without extension
#     - extension: file extension (including the dot, e.g., '.jpg')

#     Returns:
#     - A unique file name string with the given extension
#     """
#     counter = 1
#     # 组合基础文件名和扩展名
#     filename = f"{base_filename}{extension}"
#     # 如果文件名已存在，增加数字后缀直到找到未使用的名字
#     while os.path.exists(filename):
#         filename = f"{base_filename}{counter}{extension}"
#         counter += 1
#     return filename

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32,device='cuda')

def project_onto_planes(plane_axes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N,  3
    # returns projections of shape N,n_planes, 2
    """
    N, C = coordinates.shape  # N 是点的数量，C 应该是 3
    assert C == 3, "Coordinates should have 3 components (x, y, z)"
    n_planes, _, _ = plane_axes.shape #TODO rescale to triplane
    # print(f"Original coordinates shape: {coordinates.shape}")
    # print(f"Planes shape: {planes.shape}")
    # 计算 planes 的逆矩阵
    # 将 coordinates 形状扩展为 (N, n_planes, 3)
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1).reshape(N * n_planes, 3)
    inv_planes = torch.linalg.inv(plane_axes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3).to('cuda')
    # print(f"Reshaped coordinates shape: {coordinates.shape}")
    # print(f"Inverse planes shape: {inv_planes.shape}")
    # 计算投影
    # projections = torch.bmm(coordinates.unsqueeze(1), inv_planes).squeeze(1)

    projections = torch.bmm(coordinates.unsqueeze(1), inv_planes).squeeze(1)
    # print(f"Projections shape: {projections.shape}")

    # 返回形状为 (N, n_planes, 2) 的投影结果
    return projections[...,:2].view(N, n_planes, 2)
import numpy as np
def contract(x,mag):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    eps = torch.finfo(torch.float32).eps
    threshold = 0.5
    threshold_sq =threshold **2
    # mag_sq = mag **2
    # Clamping to eps prevents non-finite gradients when x == 0.
    # x_mag_sq = torch.maximum(eps, torch.sum(x**2, dim=-1, keepdim=True))
    x_mag_sq = torch.sum(x**2, dim=-1, keepdim=True)
    x_mag_sq = torch.clamp(x_mag_sq, min=eps)
    #z = torch.where(x_mag_sq <= threshold_sq *mag, x, ((torch.sqrt(mag) * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z

def sample_from_planes(plane_axes, plane_features, coordinates,max_coords,min_coords,  mode='bilinear', padding_mode='zeros', box_warp=1,radii=None):
    assert padding_mode == 'zeros'
    n_planes, C, H, W = plane_features.shape
    N,K, M = coordinates.shape#(n,K,3)
    # N, M = coordinates.shape#(n,3)

    plane_features = plane_features.view(n_planes, C, H, W)
    # coordinates = coordinates.permute(1,0,2)#K,N,3
    # print(f'sshape{coordinates.shape}')
    max_coords = project_onto_planes(plane_axes,max_coords.unsqueeze(0))
    min_coords = project_onto_planes(plane_axes,min_coords.unsqueeze(0))
    result = []
    for i in range(K):
        coordinate = coordinates[:,i,:]
        # print(f'coshape{coordinate.shape}')
        coordinate = (2/box_warp) * coordinate # TODO: add specific box bounds

        #projected_coordinates = project_onto_planes(plane_axes, coordinates)#N,3,2
        decomposed_x = torch.stack(
            [
                coordinate[:, None, [1, 2]],
                coordinate[:, None, [0, 2]],
                coordinate[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2
        projected_coordinates = decomposed_x.squeeze(2).permute(1,0,2)



        # mag_sq = torch.min(torch.norm(max_coords),torch.norm(min_coords))//2
        # mag_sq = mag_sq ** 2
        # mag_sq = torch.min(torch.sum(max_coords**2, dim=-1, keepdim=True),torch.sum(min_coords**2, dim=-1, keepdim=True))
        if radii is not None:
            mag_sq = torch.tensor(radii**2,device='cuda')
        bbxmag_sq = torch.min(torch.sum(max_coords**2, dim=-1, keepdim=True),torch.sum(min_coords**2, dim=-1, keepdim=True))#
        mag_sq=torch.min(bbxmag_sq,mag_sq)

        #normalized_coordinates = contract(projected_coordinates,mag_sq)

        normalized_coordinates =projected_coordinates/torch.sqrt(mag_sq)
        normalized_coordinates = normalized_coordinates * 2 -1 
        normalized_coordinates = contract(normalized_coordinates*6,mag_sq)/2


        normalized_coordinates = normalized_coordinates.permute(1, 0, 2).unsqueeze(2)


        # 显示图形


        # 输出特征初始化
        output_features = []

        # 对每个平面进行插值
        for i in range(plane_features.shape[0]):
            single_plane_features = plane_features[i].unsqueeze(0)  # [1, channels, H, W]
            single_plane_coordinates = normalized_coordinates[i].unsqueeze(0)   # [1,total_points, 1, 2]
            single_output_features = torch.nn.functional.grid_sample(
                single_plane_features, 
                single_plane_coordinates, 
                mode=mode, 
                padding_mode=padding_mode, 
                align_corners=False
            ).squeeze()
            output_features.append(single_output_features)

        output_features = torch.stack(output_features)  # [3, channels, total_points]
        # print(output_features.shape)

        output_features = output_features.permute(2,0,1)#N,3,channels
        result.append(output_features)
    # output_features = output_features.sum(dim=2)#TODO mean


    output = torch.stack(result,dim=0).permute(1,0,2,3)#.view(-1,K*3,C)# K,N,3,channels
    # output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(n_planes, M, C)
    return output

class Triplane(nn.Module):
    def __init__(self,feature_dim,resolution,radii):
        super().__init__()
        self.plane_axes = generate_planes()
        self.radii = radii
        compressed_dim = 8
        self.autoencoder = Autoencoder(feature_dim,resolution,compressed_dim)
        plane = torch.empty(3, feature_dim, resolution, resolution).cuda()
        torch.nn.init.uniform_(plane, -1e-2, 1e-2)
        self.compressed_plane = torch.empty(0)

        # assert plane.shape[0] == 3 and plane[1] == feature_dim
        # kernel = torch.randn((3*feature_dim,feature_dim),device='cuda')
        # bias = torch.zeros((1,feature_dim),device='cuda')
        # # self.kernel = nn.Parameter(kernel.requires_grad_(True))
        # # self.bias = nn.Parameter(bias.requires_grad_(True))
        # self.mlp =nn.Sequential(
        #     nn.Linear(3*feature_dim+3,2*feature_dim),
        #     nn.ReLU(True),
        #     nn.Linear(2*feature_dim,feature_dim)

        # ).cuda()            # nn.Linear(2*feature_dim,(feature_dim+6+3*n_offsets)*2+1+1+1)

        self.planes = nn.Parameter(plane.requires_grad_(True))

    def get_encode(self):
        #compressed_triplane, _ = self.autoencoder(self.planes)
        return self.compressed_plane

    def forward(self,sample_coordinates,max_coords,min_coords,is_training=0,step=0): 
        if is_training and step >15000:
            out = self.sample(self.planes,sample_coordinates,max_coords,min_coords)
            compressed_planes, reconstructed_planes = [], []
            for i in range(3):
                triplane_slice = self.planes[i, :, :, :].unsqueeze(0)  # 提取每个平面 (batch_size, feat, res, res) 1,feat_dim,res,res
                compressed, reconstructed = self.autoencoder(triplane_slice)
                compressed_planes.append(compressed)
                reconstructed_planes.append(reconstructed)

            # 将压缩后的和重建后的 triplane 拼接回来
            compressed_triplane = torch.stack(compressed_planes, dim=1).permute(1,0,2,3)
            reconstructed_triplane = torch.stack(reconstructed_planes, dim=1).permute(1,0,2,3)
            # compressed_triplane = torch.stack(compressed_planes, dim=0).squeeze()
            # # compressed_triplane = compressed_triplane.permute(1,0,2,3)
            # reconstructed_triplane = torch.stack(reconstructed_planes, dim=0).squeeze()
            # print(reconstructed_triplane.shape)
            # reconstructed_triplane = reconstructed_triplane.permute(1,0,2,3)
            #compressed_triplane, reconstructed_triplane = self.autoencoder(self.planes)
            # print( compressed_triplane.shape, reconstructed_triplane.shape)
            self.compressed_plane = compressed_triplane
            return out, compressed_triplane, reconstructed_triplane
        elif is_training and step<=15000:
                out = self.sample(self.planes,sample_coordinates,max_coords,min_coords)
                return out, None, None
        else:
            planes = self.planes
            out = self.sample(planes,sample_coordinates,max_coords,min_coords)
            return out


    def sample(self, planes, sample_coordinates,max_coords,min_coords):
        box_warp= 1 # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
        sampled_features = sample_from_planes(self.plane_axes, planes,sample_coordinates, max_coords, min_coords,  box_warp=box_warp,radii= 0.5*self.radii)
        # print(f'sampled_features{sampled_features.shape}')

        middle = sampled_features.reshape(sampled_features.shape[0],-1)#n,
        # output_features = middle
        # output_features = middle @ self.kernel +self.bias
        # output_features = self.mlp(torch.cat((middle,sample_coordinates),dim=1))
        # print(f'output_features{output_features.shape}')

        # return output_features
        return middle
        

class Autoencoder(nn.Module):
    def __init__(self, feat, res, compressed_dim):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=feat, out_channels=16, kernel_size=3, stride=2, padding=1),  # 降采样
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 降采样
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=compressed_dim, kernel_size=3, stride=2, padding=1),  # 降采样
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=compressed_dim, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=feat, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样到原始大小
            nn.Sigmoid()  # 输出值范围控制在0~1
        )
    
    def forward(self, x):
        # 输入张量 x 的形状为 (batch_size, feat, res, res)
        # x = x.unsqueeze(0)
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        # return compressed, reconstructed
        return compressed.squeeze(), reconstructed.squeeze()



class TriMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int = 8,
        plane_size: int = 128,
        feature_dim: int = 32,
        include_xyz: bool = False,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.include_xyz = include_xyz

        self.register_parameter(
            "fm",
            nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
        )
        self.init_parameters()
        self.dim_out = (
            self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3
        )

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level=None):
        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
        decomposed_x = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2
        if 0 == self.n_levels:
            level = None
        else:
            # assert level.shape[0] > 0, [level.shape, x.shape]
            torch.stack([level, level, level], dim=0)
            level = torch.broadcast_to(
                level, decomposed_x.shape[:3]
            ).contiguous()
        enc = nvdiffrast.torch.texture(
            self.fm,
            decomposed_x,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )  # 3xNx1xC
        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                self.feature_dim * 3,
            )
        )  # Nx(3C)
        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc