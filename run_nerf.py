import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from nerf_utils import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = inputs.reshape(-1, inputs.shape[-1])
    embedded = embed_fn(inputs_flat)

    input_dirs = viewdirs[:,None].expand(inputs.shape)
    input_dirs_flat = input_dirs.reshape(-1, input_dirs.shape[-1])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays_in_batches(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(rays_o, rays_d, chunk=1024*32, c2w=None,
          near=0., far=1.,
          **kwargs):
    """Render rays
    Args:
      rays_o: array of shape [batch_size, 3]. Ray origin for each example in batch.
      rays_d: array of shape [batch_size, 3]. Ray direction for each example in batch.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      depth_map: [batch_size]. Predicted depth values for rays.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    sh = rays_d.shape # [..., 3]
    
    # Always compute normalized view directions
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    
    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

    # Render and reshape
    all_ret = render_rays_in_batches(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None):

    H, W, focal = hwf

    rgbs = []
    depths = []
    psnrs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        t = time.time()
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rgb, depth, acc, _ = render(rays_o, rays_d, chunk=chunk, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        depths.append(depth.cpu().numpy())
        if i==0:
            print(rgb.shape, depth.shape)

        # Compute PSNR if ground truth images are provided
        if gt_imgs is not None:
            mse = np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i]))
            psnr = -10. * np.log10(mse)
            psnrs.append(psnr)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}_rgb.png'.format(i))
            imageio.imwrite(filename, rgb8)
            
            # Save depth map with matplotlib
            depth_map = depths[-1]
            plt.figure(figsize=(W/100, H/100), dpi=100)  # Match RGB image dimensions
            plt.imshow(depth_map, cmap='plasma', aspect='equal')
            plt.colorbar(label='Depth')
            plt.axis('off')
            depth_filename = os.path.join(savedir, '{:03d}_depth.png'.format(i))
            plt.savefig(depth_filename, dpi=100, bbox_inches='tight')
            plt.close()


    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    
    # Compute mean PSNR if ground truth images were provided
    mean_psnr = np.mean(psnrs) if psnrs else None

    return rgbs, depths, mean_psnr


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # Always use view directions
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=True).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=True).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and args.reload:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'white_bkgd' : args.white_bkgd,
    }

    # NDC only good for LLFF-style forward facing data
    # if args.dataset_type != 'llff' or args.no_ndc:
        # print('Not ndc!')
        # render_kwargs_train['ndc'] = False
        # render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def volume_rendering(raw, z_vals, rays_d, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
    """

    dists = compute_bin_dists(z_vals, rays_d)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    alpha = compute_alpha(raw[..., 3], dists)
    
    weights = compute_weights(alpha)
    
    #############################################################
    # TODO: 3d
    # HINT: Compute the final volume rendering outputs:
    # - rgb_map: weighted sum of colors along each ray
    # - depth_map: weighted sum of depths along each ray  
    # - acc_map: sum of weights (total opacity) along each ray
    #############################################################
    # Your code starts here
    # raise NotImplementedError("Not implemented")

    # - rgb_map: weighted sum of colors along each ray
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    # - depth_map: weighted sum of depths along each ray 
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # - acc_map: sum of weights (total opacity) along each ray
    acc_map = torch.sum(weights, dim=-1)

    # Your code ends here
    #############################################################

    if white_bkgd:
        rgb_map = rgb_map + (1 - acc_map[...,None])

    return rgb_map, depth_map, acc_map, weights


def compute_bin_dists(z_vals, rays_d):
    """Compute bin distances along a ray, matching original implementation.
    Args:
        z_vals: [N_rays, N_samples] depths along the ray
        rays_d: [N_rays, 3] ray directions
    Returns:
        dists: [N_rays, N_samples] distances used for volume rendering
    """
    #############################################################
    # TODO: Task 3a
    # HINT: Compute distances between consecutive sample points:
    # 1. Calculate differences between consecutive z_vals: z_vals[...,1:] - z_vals[...,:-1]
    # 2. Append a large value (1e10) for the last bin to represent infinite distance
    # 3. Scale by ray direction magnitude: ||rays_d|| to get actual distances in 3D space
    #############################################################
    # Your code starts here
    # raise NotImplementedError("Not implemented")

    # === 1. Calculate differences between consecutive z_vals: z_vals[...,1:] - z_vals[...,:-1] ===
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # === 2. Append a large value (1e10) for the last bin to represent infinite distance ===
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # === 3. Scale by ray direction magnitude: ||rays_d|| to get actual distances in 3D space ===
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    return dists

    # Your code ends here
    #############################################################


def compute_alpha(sigma, dists):
    """Compute alpha values for volume rendering.
    Args:
        sigma: [N_rays, N_samples] density values
        dists: [N_rays, N_samples] distances between sample points
    Returns:
        alpha: [N_rays, N_samples] alpha values for volume rendering
    """
    #############################################################
    # TODO: Task 3b
    # HINT: Compute alpha values for volume rendering:
    # alpha = 1 - exp(-sigma * delta) where
    # - sigma is the density - use ReLU to ensure non-negative
    # - delta is the distance between sample points (dists)
    #############################################################
    # Your code starts here
    # raise NotImplementedError("Not implemented")

    alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)

    return alpha

    # Your code ends here
    #############################################################


def compute_weights(alpha):
    """Compute weights for volume rendering.
    Args:
        alpha: [N_rays, N_samples] alpha values
    Returns:
        weights: [N_rays, N_samples] weights for volume rendering
    """
    #############################################################
    # TODO: Task 3c
    # HINT: Compute weights for volume rendering using alpha values:
    # Weights represent how much each sample contributes to the final color
    # You first need to compute discretized transmittance T[i] (torch.cumprod might be helpful here). Assume T[1] = 1.
    # Then weights[i] = T[i] * alpha[i]
    #############################################################
    # Your code starts here
    # raise NotImplementedError("Not implemented")

    # === 1. Compute discretized transmittance T[i] ===
    T = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )
    T = T[:, :-1]

    # === 2. Compute weights: weights[i] = T[i] * alpha[i] ===
    weights = T * alpha

    return weights

    # Your code ends here
    #############################################################


def sample_points(rays_o, rays_d, near, far, N_samples):
    """Sample coarse points along rays.
    Args:
        rays_o: [N_rays, 3] ray origins
        rays_d: [N_rays, 3] ray directions
        near: [N_rays, 1] near bounds
        far: [N_rays, 1] far bounds
        N_samples: int, number of samples per ray
    Returns:
        z_vals: [N_rays, N_samples] sampled depths along each ray
        pts: [N_rays, N_samples, 3] 3D sampled points
    """
    #############################################################
    # TODO: Task 2
    # HINT: Sample points along rays for coarse NeRF:
    # 1. Create evenly spaced bins/intervals between near and far
    # 2. Compute sampled depths by uniformly sample points in each
    #    bin for better training
    #############################################################
    # Your code starts here
    # raise NotImplementedError("Not implemented")
    N_rays = rays_o.shape[0]

    # === 1. Create evenly spaced bins/intervals between near and far ===
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(N_rays, N_samples)

    # === 2. Compute sampled depths by uniformly sample points in each bin for better training ===
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., :1], mids], dim=-1)
    rand_t = torch.rand(z_vals.shape)
    z_vals = lower + (upper - lower) * rand_t

    # Your code ends here
    #############################################################

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # pts = o + d*t

    return z_vals, pts


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                perturb=False,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: bool. If True, each ray is sampled at stratified random points
        in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      depth_map: [num_rays]. Depth map.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      depth0: See depth_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    z_vals, pts = sample_points(rays_o, rays_d, near, far, N_samples)

    # Get color and density predictions
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, depth_map, acc_map, weights = volume_rendering(raw, z_vals, rays_d, white_bkgd)

    if N_importance > 0:

        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(not perturb))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, depth_map, acc_map, weights = volume_rendering(raw, z_vals, rays_d, white_bkgd)

    ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--reload", action='store_true',
                        help='reload weights from saved ckpt (default: no reload)')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", action='store_true',
                        help='set to false for no jitter, true for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--downscale_factor", type=int, default=1, 
                        help='downscale factor for images and camera parameters (1=no downscaling, 2=half resolution)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.downscale_factor, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        # Do NOT change this to receive credit.
        i_test = i_test[[11, 17, 22]]

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.downscale_factor, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _, mean_psnr = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare training data
    N_rand = args.N_rand
    poses = torch.Tensor(poses).to(device)


    N_iters = 22000 + 1
    print('Begin')
    print(f'Number of TRAIN views are {len(i_train)}')
    print(f'Number of TEST views are {len(i_test)}')
    print(f'Number of VAL views are {len(i_val)}')

    # Clean up existing TensorBoard files
    tensorboard_dir = os.path.join(basedir, expname, 'tensorboard')
    if os.path.exists(tensorboard_dir):
        print(f"Cleaning up existing TensorBoard files in {tensorboard_dir}")
        import shutil
        shutil.rmtree(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    start = start + 1
    for i in trange(start, N_iters):
        # Random sample rays from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = render(rays_o, rays_d, chunk=args.chunk, retraw=False,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()
        

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################


        '''Rest is logging'''
        # TensorBoard logging
        writer.add_scalar('Loss/Train', loss.item(), global_step)
        if args.N_importance > 0:
            # Fine and coarse models
            writer.add_scalar('PSNR/Train_Fine', psnr.item(), global_step)
            writer.add_scalar('PSNR/Train_Coarse', psnr0.item(), global_step)
        else:
            # Only coarse model
            writer.add_scalar('PSNR/Train', psnr.item(), global_step)

        # Save checkpoints
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # Save test images
        if i in [500, 2000, 5000, 10000, 20000]:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                _, _, mean_psnr = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            if mean_psnr is not None:
                print(f"Saved test results. Mean test PSNR: {mean_psnr:.2f}")
            
            # Log test PSNR to TensorBoard
            if mean_psnr is not None:
                writer.add_scalar('PSNR/Test', mean_psnr, global_step)

        # Print training progress
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

    # Close TensorBoard writer
    writer.close()


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
