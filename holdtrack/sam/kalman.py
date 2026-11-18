"""Predict object motion for reference selection."""

import numpy as np
import scipy.linalg

"\nTable for the 0.95 quantile of the chi-square distribution with N degrees of\nfreedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv\nfunction and used as Mahalanobis gating threshold.\n"
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.07,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    def __init__(self) -> None:
        ndim, dt = (4, 1.0)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            0.01,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-05,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return (mean, covariance)

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            0.01,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-05,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        )
        return (mean, covariance)

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            0.1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return (mean, covariance + innovation_cov)

    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            0.01 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-05 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return (mean, covariance)

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return (new_mean, new_covariance)

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric="maha"):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = (mean[:2], covariance[:2, :2])
            measurements = measurements[:, :2]
        data = measurements - mean
        if metric == "gaussian":
            return np.sum(data * data, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, data.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")

    def compute_iou(self, pred_bbox, bboxes):
        ious = []
        pred_bbox = self.xyah_to_xyxy(pred_bbox)
        for bbox in bboxes:
            iou = self._compute_iou(pred_bbox, bbox)
            ious.append(iou)
        return ious

    def _compute_iou(self, bbox1, bbox2):
        if bbox2 == [0, 0, 0, 0]:
            return 0
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(
            0, min(y2, y2_) - max(y1, y1_)
        )
        union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection_area
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def xyxy_to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            h = 1
        return [xc, yc, w / h, h]

    def xyah_to_xyxy(self, bbox):
        xc, yc, a, h = bbox
        x1 = xc - a * h / 2
        y1 = yc - h / 2
        x2 = xc + a * h / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]
