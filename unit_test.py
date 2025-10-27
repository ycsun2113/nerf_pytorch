import argparse
import torch
import numpy as np
from nerf_utils import get_rays
from run_nerf import compute_bin_dists, compute_alpha, compute_weights

def test_get_rays():
    """Test cases for get_rays function"""
    print("\n==========Testing get_rays==========")
    all_correct = True

    # Case 1
    H, W = 4, 4
    focal = 2.0
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    c2w = torch.eye(4)
    
    rays_o, rays_d = get_rays(H, W, K, c2w)
    target_rays_o = torch.zeros(H, W, 3)
    target_rays_d = torch.tensor(
        [[[-1.0000,  1.0000, -1.0000],
          [-0.5000,  1.0000, -1.0000],
          [ 0.0000,  1.0000, -1.0000],
          [ 0.5000,  1.0000, -1.0000]],

         [[-1.0000,  0.5000, -1.0000],
          [-0.5000,  0.5000, -1.0000],
          [ 0.0000,  0.5000, -1.0000],
          [ 0.5000,  0.5000, -1.0000]],

         [[-1.0000,  0.0000, -1.0000],
          [-0.5000,  0.0000, -1.0000],
          [ 0.0000,  0.0000, -1.0000],
          [ 0.5000,  0.0000, -1.0000]],

         [[-1.0000, -0.5000, -1.0000],
          [-0.5000, -0.5000, -1.0000],
          [ 0.0000, -0.5000, -1.0000],
          [ 0.5000, -0.5000, -1.0000]]]
    )

    c0 = torch.allclose(rays_o, target_rays_o, rtol=1e-4, atol=1e-5)
    c1 = torch.allclose(rays_d, target_rays_d, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c0 and c1
    print(f"=== Test Case 1: Basic 4x4 image === {c0 and c1}")

    # Case 2
    H, W = 2, 2
    focal = 1.0
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    # Camera rotated 90 degrees around Y axis
    c2w = torch.tensor([
        [0, 0, 1, 1],
        [0, 1, 0, 2],
        [-1, 0, 0, 3],
        [0, 0, 0, 1]
    ]).float()

    rays_o, rays_d = get_rays(H, W, K, c2w)
    target_rays_o = torch.tensor(
        [[[1., 2., 3.],
          [1., 2., 3.]],

         [[1., 2., 3.],
          [1., 2., 3.]]]
    )
    target_rays_d = torch.tensor(
        [[[-1., 1., 1.],
          [-1., 1., 0.]],

         [[-1., 0., 1.],
          [-1., 0., 0.]]]
    )
    c0 = torch.allclose(rays_o, target_rays_o, rtol=1e-4, atol=1e-5)
    c1 = torch.allclose(rays_d, target_rays_d, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c0 and c1
    print(f"=== Test Case 2: Rotated camera === {c0 and c1}")
    
    # print("\n=== Test Case 6: Non-square image ===")
    H, W = 3, 5
    focal = 4.0
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    c2w = torch.eye(4)
    
    rays_o, rays_d = get_rays(H, W, K, c2w)
    target_rays_o = torch.zeros(H, W, 3)
    target_rays_d = torch.tensor(
        [[[-0.6250, 0.3750, -1.0000],
          [-0.3750, 0.3750, -1.0000],
          [-0.1250, 0.3750, -1.0000],
          [0.1250, 0.3750, -1.0000],
          [0.3750, 0.3750, -1.0000]],

         [[-0.6250, 0.1250, -1.0000],
          [-0.3750, 0.1250, -1.0000],
          [-0.1250, 0.1250, -1.0000],
          [0.1250, 0.1250, -1.0000],
          [0.3750, 0.1250, -1.0000]],

         [[-0.6250, -0.1250, -1.0000],
          [-0.3750, -0.1250, -1.0000],
          [-0.1250, -0.1250, -1.0000],
          [0.1250, -0.1250, -1.0000],
          [0.3750, -0.1250, -1.0000]]]
    )
    c0 = torch.allclose(rays_o, target_rays_o, rtol=1e-4, atol=1e-5)
    c1 = torch.allclose(rays_d, target_rays_d, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c0 and c1
    print(f"=== Test Case 3: Non-square image === {c0 and c1}")

    # Add your test cases here
    print(f"==========All get_rays test cases passed: {all_correct}==========")


def test_compute_bin_dists():
    """Test cases for compute_bin_dists function"""
    print("\n==========Testing compute_bin_dists==========")
    all_correct = True

    # Case 1: Basic test with simple z_vals and rays_d
    z_vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4] - single ray, 4 samples
    rays_d = torch.tensor([[1.0, 0.0, 0.0]])  # [1, 3] - ray pointing in x direction
    
    dists = compute_bin_dists(z_vals, rays_d)
    
    # Expected: differences between consecutive z_vals, scaled by ray magnitude
    # z_vals differences: [2-1, 3-2, 4-3] = [1, 1, 1]
    # Last bin should be 1e10
    # Then scale by ray magnitude: ||[1,0,0]|| = 1
    target_dists = torch.tensor([[1.0, 1.0, 1.0, 1e10]])
    
    c1 = torch.allclose(dists, target_dists, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c1
    print(f"=== Test Case 1: Basic single ray === {c1}")

    # Case 2: Multiple rays with different directions
    z_vals = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5]
    ])
    rays_d = torch.tensor([
        [1.0, 0.0, 0.0],  # magnitude = 1
        [0.0, 3.0, 4.0]   # magnitude = 5
    ])
    
    dists = compute_bin_dists(z_vals, rays_d)
    
    target_dists = torch.tensor([
        [1.0, 1.0, 1e10],
        [5.0, 5.0, 5e10]
    ])
    
    c2 = torch.allclose(dists, target_dists, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c2
    print(f"=== Test Case 2: Multiple rays with different magnitudes === {c2}")

    # Case 3: Non-uniform z_vals spacing
    z_vals = torch.tensor([[0.0, 1.0, 4.0, 9.0]])  # [1, 4] - non-uniform spacing
    rays_d = torch.tensor([[2.0, 0.0, 0.0]])  # [1, 3] - magnitude = 2
    
    dists = compute_bin_dists(z_vals, rays_d)

    target_dists = torch.tensor([[2.0, 6.0, 10.0, 2e10]])
    
    c3 = torch.allclose(dists, target_dists, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c3
    print(f"=== Test Case 3: Non-uniform z_vals spacing === {c3}")

    print(f"==========All compute_bin_dists test cases passed: {all_correct}==========")


def test_compute_alpha():
    """Test cases for compute_alpha function"""
    print("\n==========Testing compute_alpha==========")
    all_correct = True

    # Case 1: Basic test with single ray
    sigma = torch.tensor([[0.01, -0.05, 0.04, 0.2, 0.5, 0.1]])
    dists = torch.tensor([[0.03, 0.04, 0.03, 0.02, 0.01, 1e10]])
    
    alpha = compute_alpha(sigma, dists)
    target_alpha = torch.tensor([[2.9993e-04, 0, 1.1993e-03, 3.9920e-03, 4.9875e-03, 1]])
    
    c1 = torch.allclose(alpha, target_alpha, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c1
    print(f"=== Test Case 1: Basic single sample === {c1}")


    # Case 2: Basic test with multiple rays
    sigma = torch.tensor([[0.23, 0.04, 0.02, -0.12, -0.1, 0.6],
                          [-0.03, -0.15, 0.54, 0.34, 0.2, -0.31]])
    dists = torch.tensor([[0.05, 0.04, 0.03, 0.06, 0.06, 1e10],
                          [0.05, 0.04, 0.03, 0.06, 0.06, 1e10]])
    
    alpha = compute_alpha(sigma, dists)
    target_alpha = torch.tensor(
        [[1.1434e-02, 1.5987e-03, 5.9980e-04, 0, 0, 1],
        [0, 0, 1.6069e-02, 2.0193e-02, 1.1928e-02, 0]]
    )
 
    c2 = torch.allclose(alpha, target_alpha, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c2
    print(f"=== Test Case 2: Basic multiple samples === {c2}")


    print(f"==========All compute_alpha test cases passed: {all_correct}==========")


def test_compute_weights():
    """Test cases for compute_weights function"""
    print("\n==========Testing compute_weights==========")
    all_correct = True

    # Case 1: Basic test with all alpha values are 0
    alpha = torch.tensor([[0, 0, 0, 0, 0]])
    weights = compute_weights(alpha)
    target_weights = torch.zeros_like(alpha, dtype=torch.float32)
    
    c1 = torch.allclose(weights, target_weights, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c1
    print(f"=== Test Case 1: all zero alpha === {c1}")


    # Case 2: Basic test with all alpha values are 0.1
    alpha = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]])
    weights = compute_weights(alpha)
    target_weights = torch.tensor([[0.1000, 0.0900, 0.0810, 0.0729, 0.0656]])
    
    c2 = torch.allclose(weights, target_weights, rtol=1e-4, atol=1e-5)
    all_correct = all_correct and c2
    print(f"=== Test Case 2: all same values alpha === {c2}")

    print(f"==========All compute_weights test cases passed: {all_correct}==========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all', choices=['get_rays', 'compute_bin_dists', 'compute_alpha', 'compute_weights', 'all'])
    args = parser.parse_args()

    if args.test == 'get_rays':
        test_get_rays()
    elif args.test == 'compute_bin_dists':
        test_compute_bin_dists()
    elif args.test == 'compute_alpha':
        test_compute_alpha()
    elif args.test == 'compute_weights':
        test_compute_weights()
    else:
        test_get_rays()
        test_compute_bin_dists()
        test_compute_alpha()
        test_compute_weights()