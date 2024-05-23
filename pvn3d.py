import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import math
import typing


class PVN3D:

    def __init__(self):

        self.initial_pose_model = InitialPoseModel(n_point_candidate=50)
        self.base_crop = (80, 80)
        self.n_sample_points = 512

        # load custom ops
        ops_base_path = "tf_ops"
        tf.load_op_library(os.path.join(ops_base_path, "grouping", "tf_grouping_so.so"))
        tf.load_op_library(os.path.join(ops_base_path, "3d_interpolation", "tf_interpolate_so.so"))
        tf.load_op_library(os.path.join(ops_base_path, "sampling", "tf_sampling_so.so"))

        self.model_paths = {x.name: x for x in Path("weights/pvn3d").glob("*")}
        self.loaded_model = None
        self.model = None
        
        self.keypoints = {}
        for key_point_path in Path("keypoints").glob("*"):

            kpts_path = key_point_path.joinpath("farthest.txt")
            corner_path = key_point_path.joinpath("corners.txt")
            key_points = np.loadtxt(kpts_path)
            center = [np.loadtxt(corner_path).mean(0)]

            self.keypoints[key_point_path.name] = np.concatenate(
                [key_points, center], axis=0
            ).astype(np.float32)

    def inference(self, cls: str, bbox, rgb, depth, camera_matrix):
        crop_index, crop_factor = get_crop_index(
            bbox, rgb.shape[:2], base_crop_resolution=self.base_crop
        )

        x1, y1, x2, y2 = crop_index

        depth_crop = depth[y1:y2, x1:x2].copy()
        rgb_crop = rgb[y1:y2, x1:x2].copy()

        crop_index = (x1, y1)

        pcld_xyz, pcld_feat, sampled_index = pcld_processor_tf(
            depth_crop.astype(np.float32),
            rgb_crop.astype(np.float32) / 255.0,
            camera_matrix.astype(np.float32),
            tf.constant(1),
            tf.constant(self.n_sample_points),
            tf.constant(crop_index),
        )

        # print("pcld_process: ", time.perf_counter()- t_pcld)
        if pcld_xyz.shape[0] < self.n_sample_points:
            return False, None

        rgb_crop_resnet = tf.cast(tf.image.resize(rgb_crop, self.base_crop), tf.float32)

        # add batches
        rgb_crop_resnet = tf.expand_dims(rgb_crop_resnet, 0)
        pcld_xyz_ = tf.expand_dims(pcld_xyz, 0)
        pcld_feat = tf.expand_dims(pcld_feat, 0)
        sampled_index_ = tf.cast(tf.expand_dims(sampled_index, 0), tf.int32)
        crop_factor = tf.expand_dims(crop_factor, 0)

        pcld = [pcld_xyz_, pcld_feat]
        inputs = [pcld, sampled_index_, rgb_crop_resnet, crop_factor]

        if self.loaded_model != cls:
            print(f"Loading model for {cls}")
            self.model = tf.keras.models.load_model(self.model_paths[cls])
            self.loaded_model = cls

        kp_pre_ofst, seg_pre, cp_pre_ofst = self.model(inputs, training=False)

        R, t, kpts_voted, S = self.initial_pose_model(
            [pcld_xyz_, kp_pre_ofst, cp_pre_ofst, seg_pre, tf.expand_dims(self.keypoints[cls], 0)]
        )

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = tf.squeeze(t)

        return True, Rt


class InitialPoseModel:
    def __init__(self, n_point_candidate=10):
        self.n_point_candidates = n_point_candidate

    def __call__(self, inputs):
        pcld_input, kpts_pre_input, cpt_pre_input, seg_pre_input, mesh_kpts_input = inputs

        obj_kpts = self.batch_get_pt_candidates_tf(
            pcld_input, kpts_pre_input, seg_pre_input, cpt_pre_input, self.n_point_candidates
        )

        kpts_voted = self.batch_pts_clustering_with_std(obj_kpts)
        _, n_pts, _ = kpts_voted.shape
        weights_vector = tf.ones(shape=(n_pts,))
        batch_R, batch_t, S = self.batch_rt_svd_transform(
            mesh_kpts_input, kpts_voted, weights_vector
        )
        batch_t = tf.reshape(batch_t, shape=(-1, 3))  # reshape from [bs, 3, 1] to [bs, 3]
        return batch_R, batch_t, kpts_voted, S

    @tf.function
    def batch_pts_clustering_with_std(self, kps_cans, sigma=1):
        """
        filtering the points with the standard derivation in batch
        :param sigma: 3 sigma to filtering the outliers
        :param kps_cans: kps-candidates [bs, n_kp_cp, n_pts, 3]
        :return: the voted kps [bs, n_kp_cp, 3] by averaging the all key points filtered
        """

        kps_cans_transpose = tf.transpose(kps_cans, perm=(0, 1, 3, 2))  # [bs, n_kp_cp, 3, n_pts]
        # std for x y z channels [bs, n_kp_cp, 3, 1]
        std = tf.math.reduce_std(kps_cans_transpose, axis=-1, keepdims=True)
        mean = tf.math.reduce_mean(
            kps_cans_transpose, axis=-1, keepdims=True
        )  # mean for x y z channels [bs, n_kp_cp, 3, 1]
        threshold = tf.multiply(std, sigma)  # [bs, n_kp_cp, 3, 1]
        kps_mask = tf.math.abs(kps_cans_transpose - mean) < threshold  # [bs, n_kp_cp, 3, 1]
        kps_mask = tf.cast(kps_mask, dtype=tf.float32)
        kpts_filtered = tf.multiply(kps_cans_transpose, kps_mask)
        non_zeros = tf.math.count_nonzero(kpts_filtered, axis=3, dtype=kps_cans.dtype)
        new_mean = tf.math.reduce_sum(kpts_filtered, axis=3) / non_zeros  # [bs, n_kp_cp_, 3]
        return new_mean

    @tf.function
    def batch_rt_svd_transform(self, A, B, weights_vector):
        """
        Calculates the svd transform that maps corresponding points A to B in m spatial dimensions in batch
        Input:
            A: Nxm numpy array of corresponding points, usually points on mdl, dim by default: [bs, 9, 3]
            B: Nxm numpy array of corresponding points, usually points on camera axis, dim by default: [bs, 9, 3]
            centroid_A: provided centroid for partial icp
            centroid_B: provided centroid for partial icp
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        """
        # print("A_1st", A[0, :4, :])
        # print("B_1st", B[0, :4, :])

        bs, n_pts, _ = A.shape
        A_points_trans = tf.transpose(A, perm=(0, 2, 1))  # [bs, 3, n_pts]
        B_points_trans = tf.transpose(B, perm=(0, 2, 1))  # [bs, 3, n_pts]

        weights_matrix = tf.linalg.diag(weights_vector)

        num_non_zeros = tf.math.count_nonzero(weights_vector, dtype=tf.float32)

        weighted_A = tf.matmul(A_points_trans, weights_matrix)

        # print("weighted_A:", weighted_A)

        weighted_B = tf.matmul(B_points_trans, weights_matrix)

        weighted_centroid_A = (
            tf.reduce_sum(weighted_A, axis=2, keepdims=True) / num_non_zeros
        )  # [bs, 3, 1]

        # print("weighted_centroid_A:", weighted_centroid_A)

        weighted_centroid_B = (
            tf.reduce_sum(weighted_B, axis=2, keepdims=True) / num_non_zeros
        )  # [bs, 3, 1]

        center_vector_A = A_points_trans - weighted_centroid_A  # [bs, 3, n_pts]
        center_vector_B = B_points_trans - weighted_centroid_B  # [bs, 3, n_pts]

        covariance_matrix = tf.matmul(
            tf.matmul(center_vector_A, weights_matrix),
            tf.transpose(center_vector_B, perm=(0, 2, 1)),
        )  # [bs, n_pts, 3]

        # print("covariance_matrix:", covariance_matrix)

        S, U, V = tf.linalg.svd(covariance_matrix)
        det_v_ut = tf.linalg.det(tf.matmul(V, tf.transpose(U, perm=(0, 2, 1))))

        det_signs = tf.sign(det_v_ut)
        ones_vector = tf.ones(shape=(bs, 1)) * tf.expand_dims(det_signs, axis=1)
        ones_vector = tf.concat([tf.ones(shape=(bs, 2)), ones_vector], axis=1)

        mid_matrix = tf.linalg.diag(ones_vector)
        R = tf.matmul(tf.matmul(V, mid_matrix), tf.transpose(U, perm=(0, 2, 1)))
        t = weighted_centroid_B - tf.matmul(R, weighted_centroid_A)

        # print("A2B_t", t[0])
        return R, t, S

    @tf.function
    def batch_get_pt_candidates_tf(self, pcld_xyz, kpts_ofst_pre, seg_pre, ctr_ofst_pre, k=10):
        # currently k=10 is working pretty good
        """
        Applying segmentation filtering and outlier filtering
        input are batches over the same image
        :param pcld_xyz: point cloud input
        :param kpts_ofst_pre: key point offset prediction from pvn3d
        :param seg_pre: segmentation prediction from pvn3d
        :param ctr_ofst_pre: center point prediction from pvn3d
        :param ratio: the ratio of remaining points (lower norm distance)
        :return: the predicted and clustered key-points [batch, 9, 3]
        """
        k = tf.constant(k, dtype=tf.int32)
        kpts_cpts_offst_pre = tf.concat(
            [kpts_ofst_pre, ctr_ofst_pre], axis=2
        )  # [bs, n_pts, n_kp_cp, 3]
        kpts_cpts_offst_pre_perm = tf.transpose(
            kpts_cpts_offst_pre, perm=(0, 2, 1, 3)
        )  # [bs, n_kp_cp, n_pts, 3 ]

        bs, n_kp_cp, n_pts, c = kpts_cpts_offst_pre_perm.shape
        seg = tf.argmax(seg_pre, axis=-1)  # [bs, n_pts]
        seg = tf.repeat(tf.expand_dims(seg, axis=1), axis=1, repeats=n_kp_cp)
        seg = tf.repeat(tf.expand_dims(seg, axis=-1), axis=-1, repeats=c)
        seg_inv = tf.cast(tf.ones_like(seg) - seg, dtype=tf.float32)
        # [bs, n_kp_cp, n_pts]
        seg_inf = seg_inv * tf.constant(1000.0, dtype=tf.float32)

        kpts_cpts_offst_inf = (
            kpts_cpts_offst_pre_perm + seg_inf
        )  # background points will have large distance
        kpts_cpts_offst_pre_perm_norm = tf.linalg.norm(
            kpts_cpts_offst_inf, axis=-1
        )  # [bs, n_kp_cp, n_pts]
        # [bs, n_kp_cp, k]
        _, indices = tf.math.top_k(-1 * kpts_cpts_offst_pre_perm_norm, k=k)
        offst_selected = tf.gather(
            kpts_cpts_offst_pre_perm, indices, batch_dims=2
        )  # [bs, n_kp_cp, k, c]
        pcld_repeats = tf.repeat(
            tf.expand_dims(pcld_xyz, axis=1), axis=1, repeats=n_kp_cp
        )  # [bs, n_kp_cp, n_pts, c]
        pcld_repeats_selected = tf.gather(pcld_repeats, indices, batch_dims=2)

        kpts_cpts_can = pcld_repeats_selected + offst_selected
        return kpts_cpts_can


def get_crop_index(bbox, rgb_size=(480, 640), base_crop_resolution=(160, 160)):
    """
    get the crop index on original images according to the predicted bounding box
    params: bbox: predicted bbox
            rgb_size: height and width of the original input image
            target_resolution: the resolution of the target cropped images
    return:
            crop_index(left upper corner and right bottom corner) on the original image
    """
    ori_img_h, ori_img_w = rgb_size[:2]

    coor = np.array(bbox[:4], dtype=np.int32)
    (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

    x_c = (x1 + x2) * 0.5
    y_c = (y1 + y2) * 0.5
    h_t, w_t = base_crop_resolution


    bbox_w, bbox_h = (x2 - x1), (y2 - y1)
    w_factor = bbox_w / base_crop_resolution[0]
    h_factor = bbox_h / base_crop_resolution[1]
    crop_factor = math.ceil(max(w_factor, h_factor))

    w_t *= crop_factor
    h_t *= crop_factor

    x1_new = x_c - w_t * 0.5
    x2_new = x_c + w_t * 0.5
    y1_new = y_c - h_t * 0.5
    y2_new = y_c + h_t * 0.5

    if x1_new < 0:
        x1_new = 0
        x2_new = x1_new + w_t

    if x2_new > ori_img_w:
        x2_new = ori_img_w
        x1_new = x2_new - w_t

    if y1_new < 0:
        y1_new = 0
        y2_new = y1_new + h_t

    if y2_new > ori_img_h:
        y2_new = ori_img_h
        y1_new = y2_new - h_t

    return np.array([x1_new, y1_new, x2_new, y2_new]).astype(dtype=np.int), crop_factor


@tf.function
def compute_normal_map(depth, camera_matrix):
    kernel = np.array([[[[0.5, 0.5]], [[-0.5, 0.5]]], [[[0.5, -0.5]], [[-0.5, -0.5]]]])

    diff = tf.nn.conv2d(depth, kernel, 1, "VALID")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    scale_depth = tf.concat([depth / fx, depth / fy], -1)

    # clip=tf.constant(1)
    # diff = tf.clip_by_value(diff, -clip, clip)
    diff = diff / scale_depth[:, :-1, :-1, :]  # allow nan -> filter later

    mask = tf.logical_and(~tf.math.is_nan(diff), tf.abs(diff) < 5)

    diff = tf.where(mask, diff, 0.0)

    smooth = tf.constant(4)
    kernel2 = tf.cast(tf.tile([[1 / tf.pow(smooth, 2)]], (smooth, smooth)), tf.float32)
    kernel2 = tf.expand_dims(tf.expand_dims(kernel2, axis=-1), axis=-1)
    kernel2 = kernel2 * tf.eye(2, batch_shape=(1, 1))
    diff2 = tf.nn.conv2d(diff, kernel2, 1, "VALID")

    mask_conv = tf.nn.conv2d(tf.cast(mask, tf.float32), kernel2, 1, "VALID")

    diff2 = diff2 / mask_conv

    ones = tf.expand_dims(tf.ones(diff2.shape[:3]), -1)
    v_norm = tf.concat([diff2, ones], axis=-1)

    v_norm, _ = tf.linalg.normalize(v_norm, axis=-1)
    v_norm = tf.where(~tf.math.is_nan(v_norm), v_norm, [0])

    v_norm = -tf.image.resize_with_crop_or_pad(
        v_norm, depth.shape[1], depth.shape[2]
    )  # pad and flip (towards cam)
    return v_norm

@tf.function
def compute_normals(depth, camera_matrix):
    depth = tf.expand_dims(tf.expand_dims(depth, axis=-1), axis=0)
    normal_map = compute_normal_map(depth, camera_matrix)
    normals = tf.reshape(normal_map[0], (-1, 3))  # reshape als list of normals
    return normals


@tf.function
def dpt_2_cld_tf(dpt, cam_scale, cam_intrinsic, xy_offset=(0, 0), depth_trunc=2.0):
    """
    This function converts 2D depth image into 3D point cloud according to camera intrinsic matrix
    :param dpt: the 2d depth image
    :param cam_scale: scale converting units in meters
    :param cam_intrinsic: camera intrinsic matrix
    :param xy_offset: the crop left upper corner index on the original image

    P(X,Y,Z) = (inv(K) * p2d) * depth
    where:  P(X, Y, Z): the 3D points
            inv(K): the inverse matrix of camera intrinsic matrix
            p2d: the [ u, v, 1].T the pixels in the image
            depth: the pixel-wise depth value
    """

    h_depth = tf.shape(dpt)[0]
    w_depth = tf.shape(dpt)[1]

    y_map, x_map = tf.meshgrid(
        tf.range(w_depth, dtype=tf.float32), tf.range(h_depth, dtype=tf.float32)
    )  # vice versa than mgrid

    x_map = x_map + tf.cast(xy_offset[1], tf.float32)
    y_map = y_map + tf.cast(xy_offset[0], tf.float32)

    msk_dp = tf.math.logical_and(dpt > 1e-6, dpt < depth_trunc)
    msk_dp = tf.reshape(msk_dp, (-1,))

    pcld_index = tf.squeeze(tf.where(msk_dp))

    dpt_mskd = tf.expand_dims(tf.gather(tf.reshape(dpt, (-1,)), pcld_index), -1)
    xmap_mskd = tf.expand_dims(tf.gather(tf.reshape(x_map, (-1,)), pcld_index), -1)
    ymap_mskd = tf.expand_dims(tf.gather(tf.reshape(y_map, (-1,)), pcld_index), -1)

    pt2 = dpt_mskd / tf.cast(cam_scale, dpt_mskd.dtype)  # z
    cam_cx, cam_cy = cam_intrinsic[0][2], cam_intrinsic[1][2]
    cam_fx, cam_fy = cam_intrinsic[0][0], cam_intrinsic[1][1]

    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    pcld = tf.concat((pt0, pt1, pt2), axis=1)

    return pcld, pcld_index

@tf.function
def pcld_processor_tf(
    depth, rgb, camera_matrix, camera_scale, n_sample_points, xy_ofst=(0, 0), depth_trunc=2.0
):
    points, valid_inds = dpt_2_cld_tf(
        depth, camera_scale, camera_matrix, xy_ofst, depth_trunc=depth_trunc
    )
    n_valid_inds = tf.shape(valid_inds)[0]
    sampled_inds = tf.range(n_valid_inds)

    if n_valid_inds < 10:
        # because tf.function: return same dtypes
        return tf.constant([0.0]), tf.constant([0.0]), tf.constant([0], valid_inds.dtype)

    if n_valid_inds >= n_sample_points:
        sampled_inds = tf.random.shuffle(sampled_inds)

    else:
        repeats = tf.cast(tf.math.ceil(n_sample_points / n_valid_inds), tf.int32)
        sampled_inds = tf.tile(sampled_inds, [repeats])

    sampled_inds = sampled_inds[:n_sample_points]

    final_inds = tf.gather(valid_inds, sampled_inds)

    points = tf.gather(points, sampled_inds)

    rgbs = tf.reshape(rgb, (-1, 3))
    rgbs = tf.gather(rgbs, final_inds)

    normals = compute_normals(depth, camera_matrix)
    normals = tf.gather(normals, final_inds)

    feats = tf.concat([rgbs, normals], 1)
    return points, feats, final_inds
