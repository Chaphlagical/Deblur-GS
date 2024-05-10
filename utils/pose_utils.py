import torch
import numpy as np
import scipy
from pytorch3d.transforms.se3 import se3_exp_map, se3_log_map


class Pose:
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new


class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w):  # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
            ..., None, None
        ] % np.pi  # ln(R) will explode if theta==pi
        lnR = (
            1 / (2 * self.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))
        )  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack(
            [
                torch.stack([O, -w2, w1], dim=-1),
                torch.stack([w2, O, -w0], dim=-1),
                torch.stack([-w1, w0, O], dim=-1),
            ],
            dim=-2,
        )
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


class Quaternion:
    def q_to_R(self, q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qb, qc, qd, qa = q.unbind(dim=-1)
        R = torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * (qc**2 + qd**2),
                        2 * (qb * qc - qa * qd),
                        2 * (qa * qc + qb * qd),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (qb * qc + qa * qd),
                        1 - 2 * (qb**2 + qd**2),
                        2 * (qc * qd - qa * qb),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (qb * qd - qa * qc),
                        2 * (qa * qb + qc * qd),
                        1 - 2 * (qb**2 + qc**2),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        return R

    # def R_to_q(self, R, eps=1e-8):  # [B,3,3]
    #     # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    #     # FIXME: this function seems a bit problematic, need to double-check
    #     row0, row1, row2 = R.unbind(dim=-2)
    #     R00, R01, R02 = row0.unbind(dim=-1)
    #     R10, R11, R12 = row1.unbind(dim=-1)
    #     R20, R21, R22 = row2.unbind(dim=-1)
    #     t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    #     r = (1 + t + eps).sqrt()
    #     qa = 0.5 * r
    #     qb = (R21 - R12).sign() * 0.5 * (1 + R00 - R11 - R22 + eps).sqrt()
    #     qc = (R02 - R20).sign() * 0.5 * (1 - R00 + R11 - R22 + eps).sqrt()
    #     qd = (R10 - R01).sign() * 0.5 * (1 - R00 - R11 + R22 + eps).sqrt()
    #     q = torch.stack([qa, qb, qc, qd], dim=-1)
    #     for i, qi in enumerate(q):
    #         if torch.isnan(qi).any():
    #             K = (
    #                 torch.stack(
    #                     [
    #                         torch.stack(
    #                             [R00 - R11 - R22, R10 + R01, R20 + R02, R12 - R21],
    #                             dim=-1,
    #                         ),
    #                         torch.stack(
    #                             [R10 + R01, R11 - R00 - R22, R21 + R12, R20 - R02],
    #                             dim=-1,
    #                         ),
    #                         torch.stack(
    #                             [R20 + R02, R21 + R12, R22 - R00 - R11, R01 - R10],
    #                             dim=-1,
    #                         ),
    #                         torch.stack(
    #                             [R12 - R21, R20 - R02, R01 - R10, R00 + R11 + R22],
    #                             dim=-1,
    #                         ),
    #                     ],
    #                     dim=-2,
    #                 )
    #                 / 3.0
    #             )
    #             K = K[i]
    #             eigval, eigvec = torch.linalg.eigh(K)
    #             V = eigvec[:, eigval.argmax()]
    #             q[i] = torch.stack([V[3], V[0], V[1], V[2]])
    #     return q

    def q_to_Q(self, q):
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        Q_0 = torch.stack([w, -z, y, x], -1).unsqueeze(-2)
        Q_1 = torch.stack([z, w, -x, y], -1).unsqueeze(-2)
        Q_2 = torch.stack([-y, x, w, z], -1).unsqueeze(-2)
        Q_3 = torch.stack([-x, -y, -z, w], -1).unsqueeze(-2)
        Q_ = torch.cat([Q_0, Q_1, Q_2, Q_3], -2)
        return Q_

    def invert(self, q):
        qa, qb, qc, qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1, keepdim=True)
        q_inv = torch.stack([qa, -qb, -qc, -qd], dim=-1) / norm**2
        return q_inv

    def product(self, q1, q2):  # [B,4]
        q1a, q1b, q1c, q1d = q1.unbind(dim=-1)
        q2a, q2b, q2c, q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack(
            [
                q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d,
                q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c,
                q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b,
                q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a,
            ],
            dim=-1,
        )
        return hamil_prod

    def conjugate(self, q):
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        q_conj_ = torch.stack([-x, -y, -z, w], -1)
        return q_conj_

    def exp_r2q(self, x, y, z, theta):
        lambda_ = torch.sin(theta) / (2.0 * theta)
        qx = lambda_ * x
        qy = lambda_ * y
        qz = lambda_ * z
        qw = torch.cos(theta)
        return torch.stack([qx, qy, qz, qw], -1)

    def exp_r2q_taylor(self, x, y, z, theta):
        qx = (1.0 / 2.0 - 1.0 / 12.0 * theta**2 - 1.0 / 240.0 * theta**4) * x
        qy = (1.0 / 2.0 - 1.0 / 12.0 * theta**2 - 1.0 / 240.0 * theta**4) * y
        qz = (1.0 / 2.0 - 1.0 / 12.0 * theta**2 - 1.0 / 240.0 * theta**4) * z
        qw = 1.0 - 1.0 / 2.0 * theta**2 + 1.0 / 24.0 * theta**4
        return torch.stack([qx, qy, qz, qw], -1)

    def exp_r2q_parallel(self, r, eps=1e-9):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        theta = 0.5 * torch.sqrt(x**2 + y**2 + z**2+eps)
        bool_criterion = (theta < eps).unsqueeze(-1).repeat(4)
        return torch.where(
            bool_criterion, self.exp_r2q_taylor(
                x, y, z, theta), self.exp_r2q(x, y, z, theta)
        )

    def log_q2r_parallel(self, q, eps_theta=1e-20, eps_w=1e-10):
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        theta = torch.sqrt(x**2 + y**2 + z**2+eps_theta)

        bool_theta_0 = theta < eps_theta
        bool_w_0 = torch.abs(w) < eps_w
        bool_w_0_left = torch.logical_and(bool_w_0, w < 0)

        lambda_ = torch.where(
            bool_w_0,
            torch.where(
                bool_w_0_left,
                self.log_q2r_lim_w_0_left(theta),
                self.log_q2r_lim_w_0_right(theta)
            ),
            torch.where(
                bool_theta_0,
                self.log_q2r_taylor_theta_0(w, theta),
                self.log_q2r(w, theta)
            ),
        )

        r_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z], -1)

        return r_

    def log_q2r(self, w, theta):
        return 2.0 * (torch.arctan(theta / w)) / theta

    def log_q2r_taylor_theta_0(self, w, theta):
        return 2.0 / w - 2.0 / 3.0 * (theta**2) / (w * w * w)

    def log_q2r_lim_w_0_left(self, theta):
        return -torch.pi / theta

    def log_q2r_lim_w_0_right(self, theta):
        return torch.pi / theta


def se3_2_qt_parallel(wu):
    w, u = wu.split([3, 3], dim=-1)
    wx = Lie().skew_symmetric(w)
    theta = w.norm(dim=-1)
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    # A = taylor_A(theta)
    B = Lie().taylor_B(theta)
    C = Lie().taylor_C(theta)
    # R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    t = V @ u
    q = Quaternion().exp_r2q_parallel(w)
    return q, t.squeeze(-1)


def interpolation_linear(start_pos, end_pos, alpha):
    if alpha == 0:
        alpha += 0.000001
    elif alpha == 1:
        alpha -= 0.000001

    q_start, t_start = se3_2_qt_parallel(start_pos)
    q_end, t_end = se3_2_qt_parallel(end_pos)

    t_t = (1 - alpha) * t_start + alpha * t_end
    q_tau_0 = Quaternion().q_to_Q(
        Quaternion().conjugate(q_start)) @ q_end
    r = alpha * Quaternion().log_q2r_parallel(q_tau_0.squeeze(-1))
    q_t_0 = Quaternion().exp_r2q_parallel(r)
    q_t = Quaternion().q_to_Q(q_start) @ q_t_0

    R = Quaternion().q_to_R(q_t.squeeze(dim=-1))
    t = t_t.unsqueeze(dim=-1)

    poses = torch.cat([R, t], -1)
    poses = poses.reshape([3, 4])

    return poses


def interpolation_spline(pos0, pos1, pos2, pos3, alpha):
    if alpha == 0:
        alpha += 0.000001
    elif alpha == 1:
        alpha -= 0.000001

    q0, t0 = se3_2_qt_parallel(pos0)
    q1, t1 = se3_2_qt_parallel(pos1)
    q2, t2 = se3_2_qt_parallel(pos2)
    q3, t3 = se3_2_qt_parallel(pos3)

    u = alpha
    uu = alpha**2
    uuu = alpha**3
    one_over_six = 1.0 / 6.0
    half_one = 0.5

    # t
    coeff0 = one_over_six - half_one * u + half_one * uu - one_over_six * uuu
    coeff1 = 4 * one_over_six - uu + half_one * uuu
    coeff2 = one_over_six + half_one * u + half_one * uu - half_one * uuu
    coeff3 = one_over_six * uuu

    # spline t
    t_t = coeff0 * t0 + coeff1 * t1 + coeff2 * t2 + coeff3 * t3

    # R
    coeff1_r = 5 * one_over_six + half_one * u - half_one * uu + one_over_six * uuu
    coeff2_r = one_over_six + half_one * u + half_one * uu - 2 * one_over_six * uuu
    coeff3_r = one_over_six * uuu

    # spline R
    q_01 = Quaternion().q_to_Q(
        Quaternion().conjugate(q0)) @ q1  # [1]
    q_12 = Quaternion().q_to_Q(
        Quaternion().conjugate(q1)) @ q2  # [2]
    q_23 = Quaternion().q_to_Q(
        Quaternion().conjugate(q2)) @ q3  # [3]

    r_01 = Quaternion().log_q2r_parallel(q_01.squeeze(-1)) * coeff1_r  # [4]
    r_12 = Quaternion().log_q2r_parallel(q_12.squeeze(-1)) * coeff2_r  # [5]
    r_23 = Quaternion().log_q2r_parallel(q_23.squeeze(-1)) * coeff3_r  # [6]

    q_t_0 = Quaternion().exp_r2q_parallel(r_01)  # [7]
    q_t_1 = Quaternion().exp_r2q_parallel(r_12)  # [8]
    q_t_2 = Quaternion().exp_r2q_parallel(r_23)  # [9]

    q_product1 = Quaternion().q_to_Q(q_t_1) @ q_t_2  # [10]
    q_product2 = Quaternion().q_to_Q(q_t_0) @ q_product1  # [10]
    q_t = Quaternion().q_to_Q(q0) @ q_product2  # [10]

    R = Quaternion().q_to_R(q_t.squeeze(-1))

    t = t_t.unsqueeze(dim=-1)

    pose_spline = torch.cat([R, t], -1)

    poses = pose_spline.reshape([3, 4])

    return poses


def interpolation_bezier(pos, alpha):
    order = pos.shape[0]-1
    binom_coeff = [scipy.special.binom(
        order, k) for k in range(order+1)]
    # Build coefficient matrix.
    bezier_coeff = []
    for i in range(order+1):
        coeff_i = binom_coeff[i] * pow(1-alpha, order-i)*pow(alpha, i)
        bezier_coeff.append(coeff_i)
    bezier_coeff = torch.tensor(bezier_coeff).float().cuda()
    weighted_control_points = bezier_coeff.unsqueeze(-1) * pos
    weighted_control_points_mat = se3_exp_map(weighted_control_points)
    weighted_control_points_mat = weighted_control_points_mat.reshape(
        order+1, 4, 4)
    weighted_control_points_mat = weighted_control_points_mat.transpose(-1, -2)
    delta_poses_mat = torch.eye(4).float().cuda()
    for i in range(order+1):
        delta_poses_mat = torch.matmul(
            weighted_control_points_mat[i, :], delta_poses_mat)
    return delta_poses_mat[:3, ...]
