#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat
import time
import os.path

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer#,ExtendedSettings
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
from utils.loss_utils import l1_loss
K =4
def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,step=0):
    ## view frustum filtering for acceleration   
    time_sub = 0 
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    mask_anchor = pc.get_mask_anchor[visible_mask]
    mask_anchor_bool = mask_anchor.to(torch.bool)
    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()
    #tri_feat= None
    
    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    lae = None
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.3
    
 
    if is_training:

        if step> 3000 and step <= 10000:
            feat = feat + torch.empty_like(feat).uniform_( -0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets
        if step == 10000:
            pc.updatebbox()
        if step == 15000:
            pc.init_knn_indice(K)
        if step > 10000:
            if step > 15000:
                knnanchor = pc.knnanchor#anchor[knn_indices]#N,K,3
            else:
                knnanchor = anchor.unsqueeze(1).repeat(1,K,1)
                
            feat_context,compressed_triplane, reconstructed_triplane = pc.triplane(knnanchor,pc.x_bound_max,pc.x_bound_min,is_training,step=step)
            if reconstructed_triplane is not None:
                lae = l1_loss(pc.triplane.planes, reconstructed_triplane)
            feat_context = torch.cat([feat_context,anchor],dim=1)
            feat_context = pc.get_tri_mlp(feat_context)
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)
    
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)
    
            choose_idx = torch.rand_like(anchor[:, 0]) <= 0.15
            choose_idx = choose_idx & mask_anchor_bool
            feat_chosen = feat[choose_idx]
            grid_scaling_chosen = grid_scaling[choose_idx]
            grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3*pc.n_offsets)
            mean = mean[choose_idx]
            scale = scale[choose_idx]
            mean_scaling = mean_scaling[choose_idx]
            scale_scaling = scale_scaling[choose_idx]
            mean_offsets = mean_offsets[choose_idx]
            scale_offsets = scale_offsets[choose_idx]
            Q_feat = Q_feat[choose_idx]
            Q_scaling = Q_scaling[choose_idx]
            Q_offsets = Q_offsets[choose_idx]
    
            binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1,1, 3).view(-1, 3*pc.n_offsets)
            
            bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
            bit_offsets = bit_offsets * binary_grid_masks_chosen
            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate

    elif not pc.decoded_version:
        torch.cuda.synchronize(); t1 = time.time()
        knnanchor = anchor.unsqueeze(1).repeat(1,K,1)
        feat_context = pc.triplane(knnanchor,pc.x_bound_max,pc.x_bound_min,is_training)
        feat_context = torch.cat([feat_context,anchor],dim=1)

        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(pc.get_tri_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
        torch.cuda.synchronize(); 
        time_sub = time.time() - t1

    else:
        pass


    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]


    neural_opacity = neural_opacity.reshape([-1, 1])#n*k,1
    neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)

    mask = (neural_opacity>0.0)

    mask = mask.view(-1)
    opacity = neural_opacity[mask]
    binary_grid_masks = binary_grid_masks.view(-1, 1)[mask]


    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    # scaling = scaling * binary_grid_masks
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    # print(f'color{color[0]} opacity{opacity[0]} scale_rot{scale_rot[0]}')
    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param,lae
    else:
        return xyz, color, opacity, scaling, rot,time_sub

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False,step=0,splat_args = None, render_depth: bool = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_opacity_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param, lae = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training,step=step)
    else:
        xyz, color, opacity, scaling, rot,time_sub = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        # inv_viewprojmatrix=viewpoint_camera.full_proj_transform_inverse,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        # settings=splat_args,
        # render_depth = render_depth,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                "autoencoder":lae,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,

                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,splat_args = None, render_depth: bool = False,override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        #  inv_viewprojmatrix=viewpoint_camera.full_proj_transform_inverse,
        campos=viewpoint_camera.camera_center,

        sh_degree=1,
        # settings=splat_args,
        # render_depth = render_depth,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor
    opacities = pc.get_opacity


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        # opacities=opacities,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
