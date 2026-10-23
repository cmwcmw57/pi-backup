# ================================================================
# registration.py
# 설명: ORB/사용자 초기값을 robust depth ICP로 정제하고 양방향·관측성으로 검증한다.
# 사용법: from ksrc_runtime.handheld_mapping.registration import register_clouds
# ================================================================
import copy
import math

import numpy as np
import open3d as o3d

from ksrc_runtime.handheld_mapping.reconstruct import transform


REGISTRATION_SETTINGS = {
    'voxel_scales': [2., 1., .5], 'correspondence_voxel_multiplier': 3.,
    'minimum_correspondences': 100, 'minimum_fitness': .30, 'maximum_rmse_m': .02,
    'maximum_translation_correction_m': .5, 'maximum_rotation_correction_deg': 20.,
    'minimum_observability_ratio': 1e-4, 'robust_kernel': 'Huber',
    'information_semantics': 'Open3D correspondence information; not calibrated sensor covariance',
}


def correction_size(initial: np.ndarray, refined: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(initial) @ refined
    angle = math.degrees(math.acos(float(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1))))
    return float(np.linalg.norm(delta[:3, 3])), angle


def _metrics(result: o3d.pipelines.registration.RegistrationResult) -> dict:
    return {'fitness': float(result.fitness), 'inlier_rmse': float(result.inlier_rmse),
            'correspondences': len(result.correspondence_set)}


def _observability(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                   result: o3d.pipelines.registration.RegistrationResult) -> dict:
    pairs = np.asarray(result.correspondence_set)
    if len(pairs) == 0:
        return {'ratio': 0., 'eigenvalues': [0.]*6, 'reason': 'no_correspondences'}
    points = np.asarray(source.points)[pairs[:, 0]] @ result.transformation[:3, :3].T + result.transformation[:3, 3]
    normals = np.asarray(target.normals)[pairs[:, 1]]
    centered = points-points.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if radius < 1e-6 or not np.isfinite(normals).all():
        return {'ratio': 0., 'eigenvalues': [0.]*6, 'reason': 'invalid_geometry'}
    jacobian = np.column_stack((np.cross(centered/radius, normals), normals))
    eigenvalues = np.linalg.eigvalsh(jacobian.T @ jacobian/len(points))
    ratio = float(max(0., eigenvalues[0])/max(eigenvalues[-1], 1e-20))
    return {'ratio': ratio, 'eigenvalues': eigenvalues.tolist(),
            'normalization_radius_m': radius, 'reason': 'observable' if ratio >= 1e-4 else 'degenerate_geometry'}


def register_clouds(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                    initial: np.ndarray, voxel_size: float = .02) -> dict:
    """source local→target local 변환을 반환하며 거부한 변환은 적용하지 않는다."""
    result = {'accepted': False, 'reason': '', 'transformation': np.eye(4).tolist(),
              'initial_transformation': np.eye(4).tolist(), 'information': np.zeros((6, 6)).tolist(),
              'forward': {'fitness': 0., 'inlier_rmse': 0., 'correspondences': 0},
              'backward': {'fitness': 0., 'inlier_rmse': 0., 'correspondences': 0},
              'observability': {'ratio': 0., 'reason': 'not_evaluated'},
              'information_semantics': REGISTRATION_SETTINGS['information_semantics']}
    try:
        estimate = transform(initial)
    except (ValueError, TypeError):
        return dict(result, reason='invalid_initial_transform')
    if not math.isfinite(voxel_size) or not .002 <= voxel_size <= .1:
        raise ValueError('invalid registration voxel size')
    result.update(transformation=estimate.tolist(), initial_transformation=estimate.tolist())
    if any(len(cloud.points) < 100 or not np.isfinite(np.asarray(cloud.points)).all() for cloud in (source, target)):
        return dict(result, reason='insufficient_or_invalid_points')
    registration = o3d.pipelines.registration
    for factor in REGISTRATION_SETTINGS['voxel_scales']:
        voxel = voxel_size*factor
        src, dst = (copy.deepcopy(cloud).voxel_down_sample(voxel) for cloud in (source, target))
        for cloud in (src, dst):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=30))
        estimate = registration.registration_icp(
            src, dst, 3*voxel, estimate,
            registration.TransformationEstimationPointToPlane(registration.HuberLoss(max(.02, voxel*2))),
            registration.ICPConvergenceCriteria(max_iteration=60)).transformation
    try:
        estimate = transform(estimate)
    except ValueError:
        return dict(result, reason='invalid_refined_transform')
    forward = registration.evaluate_registration(src, dst, 1.5*voxel_size, estimate)
    backward = registration.evaluate_registration(dst, src, 1.5*voxel_size, np.linalg.inv(estimate))
    # evaluate_registration stores the candidate transformation as well as pairs.
    observability = _observability(src, dst, forward)
    reverse_observability = _observability(dst, src, backward)
    observability['backward_ratio'] = reverse_observability['ratio']
    observability['ratio'] = min(observability['ratio'], reverse_observability['ratio'])
    result.update(transformation=estimate.tolist(), forward=_metrics(forward), backward=_metrics(backward),
                  observability=observability, source_points=len(src.points), target_points=len(dst.points))
    diagnostic = registration.registration_icp(
        src, dst, 1.5*voxel_size, initial, registration.TransformationEstimationPointToPoint(),
        registration.ICPConvergenceCriteria(max_iteration=30))
    result['point_to_point_diagnostic'] = dict(_metrics(diagnostic),
                                             transformation=diagnostic.transformation.tolist())
    displacement, angle = correction_size(initial, estimate)
    result.update(translation_correction_m=displacement, rotation_correction_deg=angle)
    metrics = (result['forward'], result['backward'])
    reason = ('insufficient_correspondences' if min(item['correspondences'] for item in metrics) < 100 else
              'low_overlap' if min(item['fitness'] for item in metrics) < .30 else
              'high_rmse' if max(item['inlier_rmse'] for item in metrics) > .02 else
              'prior_correction_exceeded' if displacement > .5 or angle > 20 else
              'unobservable' if observability['ratio'] < 1e-4 else 'accepted')
    result.update(accepted=reason == 'accepted', reason=reason)
    if result['accepted']:
        result['information'] = registration.get_information_matrix_from_point_clouds(
            src, dst, 1.5*voxel_size, estimate).tolist()
    return result
