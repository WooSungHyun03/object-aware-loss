# adapted from https://github.com/jzhangbs/DTUeval-python
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
from pathlib import Path

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def iter_search_roots(path_like):
    path = Path(path_like).expanduser()
    if path.is_file():
        yield path.parent.resolve()
        for parent in path.parents:
            yield parent.resolve()
    else:
        yield path.resolve()
        for parent in path.parents:
            yield parent.resolve()

def resolve_dtu_eval_paths(dataset_dir, scan):
    dataset_path = Path(dataset_dir).expanduser()
    scan = int(scan)
    gt_name = f'stl{scan:03}_total.ply'

    if not dataset_path.exists():
        raise FileNotFoundError(
            f'--dataset_dir path does not exist: {dataset_path}'
        )

    candidate_roots = []
    gt_ply_file = None

    if dataset_path.is_file():
        gt_ply_file = dataset_path.resolve()
        candidate_roots.extend(dataset_path.parents)
    else:
        candidate_roots.append(dataset_path)
        candidate_roots.extend(dataset_path.parents)
        for candidate in (
            dataset_path / gt_name,
            dataset_path / 'stl' / gt_name,
            dataset_path / 'Points' / 'stl' / gt_name,
        ):
            if candidate.is_file():
                gt_ply_file = candidate.resolve()
                break

    seen = set()
    for root in candidate_roots:
        root = root.resolve()
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)

        obs_mask_file = root / 'ObsMask' / f'ObsMask{scan}_10.mat'
        plane_file = root / 'ObsMask' / f'Plane{scan}.mat'
        if gt_ply_file is None:
            candidate = root / 'Points' / 'stl' / gt_name
            if candidate.is_file():
                gt_ply_file = candidate.resolve()

        if obs_mask_file.is_file() and plane_file.is_file() and gt_ply_file is not None and gt_ply_file.is_file():
            return obs_mask_file, plane_file, gt_ply_file

    raise FileNotFoundError(
        '--dataset_dir must point to one of the following:\n'
        '  1. Official_DTU_Dataset root containing ObsMask/ and Points/stl/\n'
        '  2. Points/stl directory containing the GT point cloud\n'
        f'  3. The GT point cloud file itself ({gt_name})\n'
        'In all cases, ObsMask/Plane files must still be reachable from the same Official_DTU_Dataset tree.'
    )

def resolve_cameras_npz_path(cameras_npz, data_path, dataset_dir, scan):
    if cameras_npz:
        cameras_path = Path(cameras_npz).expanduser()
        if not cameras_path.is_file():
            raise FileNotFoundError(f'--cameras_npz file not found: {cameras_path}')
        return cameras_path.resolve()

    scan_dir_name = f'scan{int(scan)}'
    candidate_paths = []
    seen = set()

    for seed in (data_path, dataset_dir):
        for root in iter_search_roots(seed):
            for candidate in (
                root / 'cameras.npz',
                root / scan_dir_name / 'cameras.npz',
            ):
                candidate = candidate.resolve()
                candidate_key = str(candidate)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)
                candidate_paths.append(candidate)

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate
    return None

def load_scale_mat(cameras_npz_path):
    with np.load(str(cameras_npz_path)) as camera_dict:
        if 'scale_mat_0' in camera_dict:
            return camera_dict['scale_mat_0'].astype(np.float32)

        scale_keys = sorted(key for key in camera_dict.files if key.startswith('scale_mat_'))
        if not scale_keys:
            raise KeyError(
                f'No scale_mat_* entry found in cameras file: {cameras_npz_path}'
            )
        return camera_dict[scale_keys[0]].astype(np.float32)

def transform_points_to_world(points, scale_mat):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    return points * scale_mat[0, 0] + scale_mat[:3, 3][None]

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--scan', type=int, default=1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='.',
        help='Path to Official_DTU_Dataset root, Points/stl directory, or stlXXX_total.ply GT file.',
    )
    parser.add_argument(
        '--cameras_npz',
        type=str,
        default='',
        help='Optional path to scanXX/cameras.npz used to transform normalized predictions into DTU world coordinates.',
    )
    parser.add_argument(
        '--no_auto_transform',
        action='store_true',
        help='Disable automatic coordinate transform from cameras.npz.',
    )
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    args = parser.parse_args()
    obs_mask_path, plane_path, stl_path = resolve_dtu_eval_paths(args.dataset_dir, args.scan)

    scale_mat = None
    cameras_npz_path = None
    if not args.no_auto_transform:
        cameras_npz_path = resolve_cameras_npz_path(
            args.cameras_npz,
            args.data,
            args.dataset_dir,
            args.scan,
        )
        if cameras_npz_path is not None:
            print(f'Using cameras.npz for coordinate transform: {cameras_npz_path}')
            scale_mat = load_scale_mat(cameras_npz_path)
        else:
            print('No cameras.npz found automatically. Assuming --data is already in DTU world coordinates.')

    thresh = args.downsample_density
    if args.mode == 'mesh':
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        data_mesh = o3d.io.read_triangle_mesh(args.data)

        vertices = np.asarray(data_mesh.vertices)
        if scale_mat is not None:
            vertices = transform_points_to_world(vertices, scale_mat)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)
    
    elif args.mode == 'pcd':
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(args.data)
        data_pcd = np.asarray(data_pcd_o3d.points)
        if scale_mat is not None:
            data_pcd = transform_points_to_world(data_pcd, scale_mat)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(str(obs_mask_path))
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = args.patch_size
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]
    if data_in.shape[0] == 0:
        raise ValueError(
            'No predicted points fall inside the DTU evaluation bounding box. '
            'This usually means the input mesh/pcd is not in DTU world coordinates. '
            'Pass --cameras_npz <scanXX/cameras.npz> or ensure the input was transformed before evaluation.'
        )

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]
    if data_in_obs.shape[0] == 0:
        raise ValueError(
            'No predicted points remain after DTU ObsMask filtering. '
            'The mesh/pcd and ObsMask are likely in different coordinate systems, or the scan id does not match the input.'
        )

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(str(stl_path))
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(str(plane_path))['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    
    import json
    with open(f'{args.vis_out_dir}/results.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s,
            'mean_s2d': mean_s2d,
            'overall': over_all,
        }, fp, indent=True)
