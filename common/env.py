"""
env.py — Franka Panda 穿孔任务强化学习环境
====================================
任务描述：
    Franka Panda 机械臂（7 自由度：全部关节参与控制，夹爪忽略）从初始位置出发，
    穿过竖直放置的带孔板，到达板后方的目标点。
    机器人必须对准孔洞穿越，碰撞板体则本轮终止。

对外接口（train.py / ppo.py 使用）
------------------------------------
    env = HoleBoardEnv(cfg: EnvConfig)

    scalar_obs, depth_img, info = env.reset()
        scalar_obs : np.ndarray  shape=(SCALAR_DIM,)  float32   标量观测
        depth_img  : np.ndarray  shape=(IMG_H, IMG_W)  float32  深度图 (米)
        info       : dict        {'hole_center': (y,z), 'target_pos': (x,y,z)}

    scalar_obs, depth_img, reward, done, info = env.step(action)
        action     : np.ndarray  shape=(ACTION_DIM,)  float32   已在 [-1,1]
        reward     : float
        done       : bool
        info       : dict        {'success': bool, 'collision': bool, 'timeout': bool,
                                  'ee_pos': np.ndarray, 'dist_to_target': float}

    env.close()

对外接口（更新后）：
    scalar_obs, wrist_depth, global_depth, info = env.reset()
        wrist_depth  : np.ndarray  shape=(IMG_H, IMG_W)  float32  归一化到 [0,1]
        global_depth : np.ndarray  shape=(IMG_H, IMG_W)  float32  归一化到 [0,1]

    scalar_obs, wrist_depth, global_depth, reward, done, info = env.step(action)

network.py 需导入的常量：
    SCALAR_DIM = 27        # Actor 标量观测维度：joints(15)+EE(7)+target(4)+alive(1)
    PRIVILEGED_DIM = 44    # Critic 纯标量全状态：44D，无 CNN
    IMG_H, IMG_W = 64, 64  # 深度图分辨率（像素）
    ACTION_DIM  = 5        # 动作维度（5-DOF，去掉 wrist_3）
    WRIST_DEPTH_MAX  = 2.0 # wrist 相机深度 clip 上限（米）
    GLOBAL_DEPTH_MAX = 5.0 # global 相机深度 clip 上限（米）
"""

# ── 必须第一行启动仿真 ───────────────────────────────────────────────────────
from isaacsim import SimulationApp
# headless=True 时不显示界面，训练时可设为 True 加速
# 由 EnvConfig.headless 控制，此处先用占位符，实例化时传入
# ⚠️  SimulationApp 只能在全局创建一次，train.py 里应该直接导入 sim_app 实例
# 这里做成懒初始化，由 HoleBoardEnv.__init__ 负责创建
import sys
import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# 确保项目根目录在 sys.path，无论从哪个脚本导入 env.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.normalization import Normalization

# ── 供 network.py 导入的维度常量 ──────────────────────────────────────────────
# SCALAR_DIM = sin/cos(7) + qdot(7) + ee_pos(3) + ee_quat(4)
#            + vec_target(3) + dist_target(1) + alive(1) = 33
SCALAR_DIM       = 31    # Actor 标量观测：joints(18)+EE(7)+target(4)+passed(1)+alive(1)
# PRIVILEGED_DIM = joints(21)+EE(7)+hole_center(3)+hole_normal(3)
#                + ee_to_hole(3)+ee_to_target(3)+joint_dists(7)
#                + ee_axis_angle(1)+board_pos(3)+hole_diam(1) = 52
PRIVILEGED_DIM   = 48    # Critic 纯标量特权观测，不含 CNN
IMG_H            = 64    # 深度图高度（像素）
IMG_W            = 64    # 深度图宽度（像素）
ACTION_DIM       = 6     # 6-DOF 控制（去掉 joint7 腕部旋转，忽略夹爪）
WRIST_DEPTH_MAX  = 2.0   # wrist 相机深度 clip 上限（米）
GLOBAL_DEPTH_MAX = 5.0   # global 相机深度 clip 上限（米）

# ── 关节索引 ──────────────────────────────────────────────────────────────────
ACTIVE_JOINT_IDX = [0, 1, 2, 3, 4, 5]     # Franka 手臂 6 个关节（joint7 腕部旋转固定，index 6/7/8 不控制）
N_JOINTS_TOTAL   = 9                        # Franka 总关节数（7 臂 + 2 夹爪）


# ─────────────────────────────────────────────────────────────────────────────
# 配置类（train.py 实例化并传入 HoleBoardEnv）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    # ── 仿真 ──────────────────────────────────────────────
    headless: bool  = False          # True = 无界面，加快训练
    physics_dt: float = 1.0 / 60.0  # 物理步长（秒）
    rendering_dt: float = 1.0 / 60.0

    # ── 场景几何（单位：米）───────────────────────────────
    board_x: float  = 0.1   # 板在 X 轴上的位置（固定，机械臂向 +X 方向伸出）
    board_thickness: float = 0.02   # 板厚度
    board_half_y: float = 0.8      # 板半宽（Y 方向）
    board_z_low: float  = 0.20      # 板下沿 Z 坐标
    board_z_high: float = 1.20      # 板上沿 Z 坐标

    # 孔洞随机范围（孔心在板面内的偏移量）
    hole_y_range: Tuple[float, float]  = (-0.1, 0.1)   # 孔心 Y 随机范围
    hole_z_range: Tuple[float, float]  = (0.50,  0.60)   # 孔心 Z 随机范围
    hole_half_size: float = 0.15  # 孔洞半边长（方形孔，全边长 ≈ 30 cm）

    # 目标点（板后方）
    target_x_range: Tuple[float, float]  = (0.30, 0.50)  # 目标 X 随机范围
    target_y_range: Tuple[float, float]  = (-0.05, 0.05) # 目标 Y 随机范围（相对孔心偏移）
    target_z_range: Tuple[float, float]  = (-0.05, 0.05) # 目标 Z 随机范围（相对孔心偏移，与孔同高）

    # ── 机械臂初始关节角（弧度）─────────────────────────
    # init_joints 用于设置每次开始时的机械臂的初始姿态（Franka Panda 9-DOF）
    init_joints: list = field(default_factory=lambda: [
         0.0,    # panda_joint1
        -1.7,  # panda_joint2
         0.0,    # panda_joint3
        -2.356,  # panda_joint4
         0.0,    # panda_joint5
         1.571,  # panda_joint6
         0.785,  # panda_joint7
         0.01,   # panda_finger_joint1（夹爪，微开，不参与控制）
         0.01,   # panda_finger_joint2
    ])

    # ── 动作 ──────────────────────────────────────────────
    action_scale: float = 0.20   # 将 [-1,1] 动作映射到关节增量（弧度/步）

    # ── 课程阶段（影响奖励权重，由 train.py 通过 env.set_stage() 切换）────
    stage: int = 1   # 1 / 2 / 3

    # ── 终止条件 ──────────────────────────────────────────
    max_steps: int   = 500       # 最大步数（超时）
    success_dist: float = 0.06   # EE 距目标 < 该值视为成功（穿孔后），8cm 对 stage2 足够
    workspace_limit: float = 1.5 # EE 距原点超过此值视为越界

    # ── 相机 ──────────────────────────────────────────────
    # Wrist Camera：挂载在 ee_link，随 EE 运动，近距离观察孔洞
    wrist_cam_offset: Tuple = (0.0, 0.0, 0.1)  # 中心对齐，前伸8cm补偿到接近EE末端

    # Global Camera：固定在场景，俯瞰整个工作区
    global_cam_pos:  Tuple = (1.0, 0.0, 1.2)   # 世界坐标（米）
    global_cam_look: Tuple = (0.40, 0.0, 0.55)  # 朝向目标点（指向板中心区域）

    # ── 调试 ───────────────────────────────────────────────
    debug_diagnostics: bool = False
    debug_log_episodes: int = 3
    debug_log_steps: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# 终止管理器
# ─────────────────────────────────────────────────────────────────────────────
class TerminationManager:
    """
    统一管理所有 episode 终止条件，在 step() 中调用 check()。

    终止条件：
      成功    — EE 已穿孔 且 到目标点距离 < success_dist
      硬碰撞  — 机械臂接触板体（由 _check_collision() 几何检测传入）
      超时    — step_count >= max_steps（500 steps）
      越界    — ‖ee_pos‖ > workspace_limit
    """

    def __init__(self, cfg: EnvConfig):
        self._success_dist    = cfg.success_dist
        self._max_steps       = cfg.max_steps
        self._workspace_limit = cfg.workspace_limit

    def check(
        self,
        step_count:   int,
        ee_pos:       np.ndarray,
        passed_board: bool,
        target_pos:   np.ndarray,
        collided:     bool,
    ) -> Tuple[bool, dict]:
        """
        Returns
        -------
        done      : bool
        term_info : dict  各条件触发标志 + 辅助数值
        """
        dist_to_target = float(np.linalg.norm(ee_pos - target_pos))

        success   = passed_board and (dist_to_target < self._success_dist)
        timeout   = step_count >= self._max_steps
        out_of_ws = float(np.linalg.norm(ee_pos)) > self._workspace_limit

        done = success or collided or timeout or out_of_ws

        return done, {
            "success":          success,
            "collision":        collided,
            "timeout":          timeout,
            "out_of_workspace": out_of_ws,
            "dist_to_target":   dist_to_target,
            "step_count":       step_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 奖励管理器
# ─────────────────────────────────────────────────────────────────────────────
class RewardManager:
    """
    奖励函数管理器：每个 term 作为独立方法，按课程阶段加权求和。

    各 term 说明：
      R_approach  — 穿孔前势能差，φ = -d(EE,孔心)，穿孔后冻结
      R_target    — 穿孔后势能差，φ = -d(EE,目标)，穿孔瞬间基准清零防尖峰
            R_tguide    — 穿孔前对目标点的 YZ 平面对齐引导（势能差）
      R_align     — EE 速度与 +X 轴对准，仅在距板面 < 0.3m 时激活
      R_pass      — 穿孔稀疏奖励（干净穿过 +15，擦边 +5）
      R_collision — 软接触 / 硬碰撞惩罚
      R_smooth    — 动作平滑惩罚 -||Δa||²
      R_arrive    — 到达目标点的一次性奖励

    课程阶段权重（stage 1→2→3→4 逐步收紧）：
      stage 1：放大 progress，关闭 align/smooth，降低碰撞惩罚
      stage 2：恢复正常，开启 align/smooth
      stage 3：强化 align，增大 smooth 惩罚
      stage 4：孔变小 + 目标随机性更大，强化 align/smooth，碰撞惩罚最严
    """

    _STAGE_CFG = { #将权重存为一个字典，方便根据 stage 切换
        1: dict(w_progress=10.0, w_target_guide=5.0,  w_align=0.005, w_post_align=0.0,
                r_pass_clean=5.0, r_pass_grazing=4.0,
                r_soft_coll=-0.1,  r_hard_coll=-0.5,
                w_smooth=0.0,      r_arrive=10.0,  r_alive=-0.005),

        # stage 2：过渡阶段，success_dist 放宽至 0.15m（via _STAGE_GEOM），
        # w_target_guide 桥接 Stage1(5.0) 与 Stage3(10.0)
        2: dict(w_progress=8.0,  w_target_guide=10.0,  w_align=0.005, w_post_align=0.005,
                r_pass_clean=5.0, r_pass_grazing=2.0,
                r_soft_coll=-0.2,  r_hard_coll=-0.8,
                w_smooth=0.0,      r_arrive=10.0,  r_alive=-0.005),

        # stage 3（原 Stage 2）：target 偏移收窄，w_align 加大强制垂直进场，碰撞惩罚维持
        3: dict(w_progress=8.0,  w_target_guide=12.0, w_align=0.012,  w_post_align=0.008,
                r_pass_clean=5.0, r_pass_grazing=1.0,
                r_soft_coll=-0.5,  r_hard_coll=-1.0,
                w_smooth=0.001,    r_arrive=10.0,  r_alive=-0.007),

        # stage 4（原 Stage 3）：随机化加大，碰撞惩罚最严
        4: dict(w_progress=5.0,  w_target_guide=15.0, w_align=0.015,  w_post_align=0.01,
                r_pass_clean=5.0, r_pass_grazing=0.1,
                r_soft_coll=-0.5,  r_hard_coll=-2.0,
                w_smooth=0.005,    r_arrive=10.0,  r_alive=-0.01),
    }

    def __init__(self, cfg: 'EnvConfig', stage: int = 1):
        self.cfg   = cfg
        self.stage = stage
        self._prev_phi_approach = 0.0   # φ_approach 基准，穿孔后冻结
        self._prev_phi_target   = 0.0   # φ_target   基准，穿孔瞬间重置
        self._prev_phi_tguide   = 0.0   # φ_tguide   基准（仅穿孔前有效）
        self._arrived           = False
        self._pass_rewarded     = False
        self._hole_center       = np.zeros(3, dtype=np.float32)  # 每步由 compute() 更新
        self.last_breakdown     = {"progress": 0., "align": 0., "post_align": 0., "pass": 0.,
                                   "target_guide": 0., "collision": 0., "smooth": 0., "arrive": 0., "alive": 0.}
        #last_breakdown 用于存储上一步各奖励 term 的值，供 evaluate_policy 统计用

    def set_stage(self, stage: int):
        assert stage in (1, 2, 3, 4), f"stage 必须是 1/2/3/4，got {stage}"
        self.stage = stage

    # ── episode 重置 ──────────────────────────────────────────────────────────
    def reset(self, ee_pos: np.ndarray, hole_center: np.ndarray,
              target_pos: np.ndarray):
        """每 episode 开始时调用，初始化两段势能基准和到达标志。"""
        self._prev_phi_approach = -float(np.linalg.norm(ee_pos - hole_center))
        self._prev_phi_target   = -float(np.linalg.norm(ee_pos - target_pos))
        self._prev_phi_tguide   = -float(np.linalg.norm((ee_pos - target_pos)[1:3]))
        self._arrived       = False
        self._pass_rewarded = False

    # ── 各 reward term ────────────────────────────────────────────────────────
    def _r_approach(self, ee_pos: np.ndarray, hole_center: np.ndarray,
                    passed_board: bool) -> float:
        """
        穿孔前势能差：φ_approach = -d(EE, hole_center)。
        穿孔后返回 0，基准冻结（不再更新）。
        """
        if passed_board:
            return 0.0
        phi = -float(np.linalg.norm(ee_pos - hole_center))
        r = phi - self._prev_phi_approach
        self._prev_phi_approach = phi
        return r 

    def _r_target(self, ee_pos: np.ndarray, target_pos: np.ndarray,
                  passed_board: bool, just_passed: bool) -> float:
        """
        穿孔后势能差：φ_target = -d(EE, target_pos)。
        穿孔前返回 0；穿孔瞬间（just_passed=True）将基准重置为当前值，
        本帧奖励 = 0，防止两段参考系混用产生尖峰。
        """
        if not passed_board:
            return 0.0
        phi = -float(np.linalg.norm(ee_pos - target_pos))
        if just_passed:   # 穿孔帧：重置基准，不给尖峰
            self._prev_phi_target = phi
            return 0.0
        r = phi - self._prev_phi_target
        self._prev_phi_target = phi
        return r

    def _r_align(self, ee_pos: np.ndarray, ee_vel: np.ndarray,
                 passed_board: bool = False) -> float:
        """
        cos(θ) − 1，θ 为 EE 速度与 +X 方向夹角，∈ [-2, 0]。
        仅在 EE 距板面 < 0.05m 且尚未穿孔时激活（缩短激活距离，避免 Y/Z 定位时被惩罚）。
        穿孔后关闭，避免干扰 EE 向目标点侧向移动。
        """
        if passed_board:
            return 0.0
        if abs(ee_pos[0] - self.cfg.board_x) >= 0.05:
            return 0.0
        speed = float(np.linalg.norm(ee_vel)) + 1e-8
        cos_theta = float(np.clip(ee_vel[0] / speed, -1.0, 1.0))
        return cos_theta - 1.0

    def _r_post_align(self, ee_vel: np.ndarray, ee_pos: np.ndarray,
                      target_pos: np.ndarray, passed_board: bool) -> float:
        """
        穿孔后速度对准目标方向：cos(v, target-ee) - 1 ∈ [-2, 0]。
        与 _r_align（穿孔前对准 +X）对称，逼策略穿孔后主动转向目标。
        """
        if not passed_board:
            return 0.0
        to_target = target_pos - ee_pos
        dist  = float(np.linalg.norm(to_target)) + 1e-8
        speed = float(np.linalg.norm(ee_vel))    + 1e-8
        cos_theta = float(np.clip(np.dot(ee_vel, to_target) / (speed * dist), -1.0, 1.0))
        return cos_theta - 1.0

    def _r_target_guide(self, ee_pos: np.ndarray, target_pos: np.ndarray,
                        passed_board: bool, just_passed: bool) -> float:
        """
        目标点引导奖励（仅穿孔后）：
        使用 YZ 平面到目标点的势能差，鼓励穿孔后继续侧向对齐目标。
        """
        if not passed_board:
            return 0.0
        delta_yz = (ee_pos - target_pos)[1:3] #[1,3] 是 YZ 分量
        phi = -(abs(float(delta_yz[0])) + abs(float(delta_yz[1])))  # L1 距离，越近越大 abs是为了保证是正值
        if just_passed:
            self._prev_phi_tguide = phi
            return 0.0
        r = phi - self._prev_phi_tguide 
        self._prev_phi_tguide = phi
        return r

    def _r_pass(self, just_passed: bool, ee_pos: np.ndarray) -> float:
        """
        穿孔稀疏奖励：
          干净穿过（EE 距孔边缘 >= 5cm）→ r_pass_clean (+15)
          擦边穿过（EE 距孔边缘  < 5cm）→ r_pass_grazing (+5)
        """
        if self._pass_rewarded or not just_passed:
            return 0.0
        self._pass_rewarded = True
        cfg = self.cfg
        h  = cfg.hole_half_size
        hy = self._hole_center[1]
        hz = self._hole_center[2]
        dy = abs(ee_pos[1] - hy)
        dz = abs(ee_pos[2] - hz)
        dist_to_edge = min(h - dy, h - dz)   # 正值 = 离孔边缘的余量
        w = self._STAGE_CFG[self.stage]
        return w['r_pass_grazing'] if dist_to_edge < 0.05 else w['r_pass_clean']

    def _r_collision(self, ee_pos: np.ndarray, collided: bool) -> float:
        """
        碰撞惩罚：
          硬碰撞                           → r_hard_coll（-10 / -5）
          软接触（在孔内但距孔边 < 0.02m，
                  且 EE-X 在板面 ±0.02m 内）→ r_soft_coll（-2 / -0.5）
        """
        w = self._STAGE_CFG[self.stage]
        if collided:
            return w['r_hard_coll']
        # 软接触：EE 贴近板面 且 在孔内但紧贴孔边
        cfg = self.cfg
        half_th = cfg.board_thickness / 2.0
        if abs(ee_pos[0] - cfg.board_x) > half_th + 0.02:
            return 0.0
        if not (-cfg.board_half_y < ee_pos[1] < cfg.board_half_y):
            return 0.0
        if not (cfg.board_z_low < ee_pos[2] < cfg.board_z_high):
            return 0.0
        h  = cfg.hole_half_size
        hy = self._hole_center[1]
        hz = self._hole_center[2]
        dy = abs(ee_pos[1] - hy)
        dz = abs(ee_pos[2] - hz)
        in_hole = (dy < h) and (dz < h)
        if not in_hole:
            return 0.0   # 应已被 hard collision 捕获
        dist_to_edge = min(h - dy, h - dz)
        return w['r_soft_coll'] if dist_to_edge < 0.02 else 0.0

    def _r_smooth(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """动作平滑惩罚 −||a_t − a_{t-1}||²"""
        return -float(np.sum((action - prev_action) ** 2))

    def _r_arrive(self, ee_pos: np.ndarray, target_pos: np.ndarray,
                  passed_board: bool) -> float:
        """到达目标点的一次性奖励（dist < success_dist 且已穿孔）。"""
        if self._arrived:
            return 0.0
        if passed_board and float(np.linalg.norm(ee_pos - target_pos)) < self.cfg.success_dist:
            self._arrived = True
            return 1.0
        return 0.0

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def compute(  #用于计算当前步总奖励，返回 float，并更新 self.last_breakdown 以供 info 输出
        self,
        ee_pos:       np.ndarray,
        ee_vel:       np.ndarray,
        action:       np.ndarray,
        prev_action:  np.ndarray,
        hole_center:  np.ndarray,
        target_pos:   np.ndarray,
        passed_board: bool,
        collided:     bool,
        just_passed:  bool,
    ) -> float:
        """计算当前步总奖励，返回 float。"""
        w = self._STAGE_CFG[self.stage]
        self._hole_center = hole_center   # 供 _r_pass / _r_collision 使用

        r_progress = w['w_progress'] * (
            self._r_approach(ee_pos, hole_center, passed_board) +
            self._r_target(ee_pos, target_pos, passed_board, just_passed)
        )
        r_tguide     = w['w_target_guide'] * self._r_target_guide(ee_pos, target_pos, passed_board, just_passed)
        r_align      = w['w_align']      * self._r_align(ee_pos, ee_vel, passed_board)
        r_post_align = w['w_post_align'] * self._r_post_align(ee_vel, ee_pos, target_pos, passed_board)
        r_pass     = self._r_pass(just_passed, ee_pos)        # 内含阶段值
        r_coll     = self._r_collision(ee_pos, collided)      # 内含阶段值
        r_smooth   = w['w_smooth']   * self._r_smooth(action, prev_action)
        r_arrive   = w['r_arrive']   * self._r_arrive(ee_pos, target_pos, passed_board)
        r_alive    = w['r_alive']    # 固定存活惩罚，每步扣分，逼策略尽快完成任务

        self.last_breakdown = {
            "progress":    r_progress,
            "target_guide":r_tguide,
            "align":       r_align,
            "post_align":  r_post_align,
            "pass":        r_pass,
            "collision":   r_coll,
            "smooth":      r_smooth,
            "arrive":      r_arrive,
            "alive":       r_alive,
        }
        return float(r_progress + r_tguide + r_align + r_post_align + r_pass + r_coll + r_smooth + r_arrive + r_alive)


# ─────────────────────────────────────────────────────────────────────────────
# 主环境类
# ─────────────────────────────────────────────────────────────────────────────
class HoleBoardEnv:
    """
    UR10e 穿孔任务强化学习环境（Isaac Sim 独立脚本模式）。

    使用方式（在 train.py 里）：
        cfg = EnvConfig(headless=True)
        env = HoleBoardEnv(cfg)
        obs, depth, info = env.reset()
        obs, depth, reward, done, info = env.step(action)
        env.close()
    """

    def __init__(self, cfg: EnvConfig, sim_app: Optional[SimulationApp] = None):
        """
        Parameters
        ----------
        cfg     : EnvConfig  超参数配置
        sim_app : SimulationApp  若 train.py 已创建可直接传入，避免二次初始化
                  若为 None，则此处自动创建（适合单独调试 env.py 时使用）
        """
        self.cfg = cfg

        # ── 启动仿真 App ──────────────────────────────────────────────────────
        # SimulationApp 全局只能有一个实例。
        # 正式训练时 train.py 先创建再传入；单独调试时这里自动创建。
        if sim_app is None:
            self._sim_app = SimulationApp({
                "headless": cfg.headless,
                "renderer": "RayTracedLighting",
            })
        else:
            self._sim_app = sim_app  # 使用外部传入的实例，不重复创建

        # ── 延迟导入（必须在 SimulationApp 启动后才能 import）─────────────────
        self._late_imports()

        # ── 创建 Isaac Sim World ───────────────────────────────────────────────
        from omni.isaac.core import World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # ── 运行时状态 ────────────────────────────────────────────────────────
        self._step_count    = 0           # 当前 episode 步数
        self._prev_action   = np.zeros(ACTION_DIM, dtype=np.float32)  # 上一步动作（平滑惩罚用）
        self._prev_ee_x     = 0.0        # 上一步 EE 的 X 坐标（穿孔检测用）
        self._passed_board  = False       # 是否已穿过板面
        self._episode_count = 0
        self._pending_stage = None        # 课程切换请求（下次 reset 生效）

        # 课程切换统计（滚动 30 回合）是30个episode 的窗口，记录每 episode 是否成功穿孔和到达目标，用于 evaluate_policy 统计不同课程阶段的成功率趋势
        self._curriculum_window = 30
        self._pass_history = deque(maxlen=self._curriculum_window)    # 穿孔成功：_passed_board
        self._arrive_history = deque(maxlen=self._curriculum_window)  # 到达目标：term_info["success"]

        # 当前 episode 的孔洞中心和目标坐标（每次 reset 随机化）
        self._hole_center   = np.zeros(3, dtype=np.float32)   # (x=board_x, y, z)
        self._target_pos    = np.zeros(3, dtype=np.float32)   # (x, y, z)

        # 板件的 prim 引用（_build_scene 中填充，_reset_board 中更新位置）
        self._board_pieces  = []   # list of FixedCuboid

        # ── 搭建场景 ──────────────────────────────────────────────────────────
        self._build_scene()

        # 必须在所有 scene.add() 之后调用一次 reset，让物理引擎初始化
        self.world.reset()
        self._sim_app.update()   # 驱动一帧渲染，让 OmniGraph 管线完成初始化
        self._setup_camera()     # annotator.attach() 必须在渲染管线就绪后才能调用

        # ── PhysX Tensor API（懒初始化）─────────────────────────────────────────
        # 必须在 world.step() 至少执行一次后初始化（_init_physx_art_view）。
        # link 索引也在那时从 PhysX 实际 link 名称中确定，避免 USD link 与物理 link 不一致。
        self._physics_sim_view    = None
        self._art_physx_view      = None
        self._ee_link_physx_idx   = None   # 由 _init_physx_art_view() 填充
        self._arm_link_physx_idxs = None   # 由 _init_physx_art_view() 填充

        # EE 速度估计（用于特权观测 cos(θ)）：通过相邻帧位移差分计算
        self._prev_ee_pos = np.zeros(3, dtype=np.float32)

        # ── 观测归一化器 ──────────────────────────────────────────────────────────
        self.scalar_norm = Normalization(shape=SCALAR_DIM)
        self.priv_norm   = Normalization(shape=PRIVILEGED_DIM)
        self.training    = True   # env.training=False 时归一化器不更新统计量

        # ── 终止管理器 ────────────────────────────────────────────────────────
        self._term_manager   = TerminationManager(cfg)

        # ── 奖励管理器 ────────────────────────────────────────────────────────
        self._reward_manager = RewardManager(cfg, stage=cfg.stage)

    # ──────────────────────────────────────────────────────────────────────────
    # 延迟导入（在 SimulationApp 启动后执行）
    # ──────────────────────────────────────────────────────────────────────────
    def _late_imports(self):
        """将所有 Isaac Sim / omni 相关 import 集中到这里，避免在 SimulationApp 启动前导入报错。"""
        global np  # numpy 可以提前导入，这里只是保持结构统一
        import omni.replicator.core as rep
        import omni.usd
        from omni.isaac.core.objects import FixedCuboid, VisualSphere, VisualCuboid
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.prims import XFormPrimView
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction
        from isaacsim.storage.native import get_assets_root_path
        from pxr import UsdGeom, Gf, Usd   # type: ignore  # OpenUSD 绑定，Isaac Sim 运行时始终可用
        import omni.physics.tensors.impl.api as physx
        from isaacsim.core.simulation_manager import SimulationManager

        # 保存到 self，供后续方法使用
        self._rep              = rep
        self._omni_usd         = omni.usd
        self._UsdGeom          = UsdGeom
        self._Gf               = Gf
        self._Usd              = Usd
        self._physx            = physx
        self._SimulationManager = SimulationManager
        self._FixedCuboid   = FixedCuboid
        self._VisualSphere  = VisualSphere
        self._XFormPrimView = XFormPrimView
        self._Robot         = Robot
        self._add_ref       = add_reference_to_stage
        self._ArtAction     = ArticulationAction
        self._assets_root   = get_assets_root_path()

    # ──────────────────────────────────────────────────────────────────────────
    # 搭建仿真场景（只在 __init__ 中调用一次）
    # ──────────────────────────────────────────────────────────────────────────
    def _build_scene(self):
        """
        搭建静态场景结构：
          1. UR10e 机械臂
          2. 带孔板（4 块 FixedCuboid 拼成框架，孔位在 reset 时随机化）
          3. 目标点可视化球体
        注意：板的位置在 _reset_board() 中每 episode 更新。
        """
        cfg = self.cfg

        # ── 1. 加载 Franka Panda ──────────────────────────────────────────────────
        franka_usd = self._assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        self._add_ref(usd_path=franka_usd, prim_path="/World/Franka")

        # Robot 封装：提供 get_joint_positions / apply_action 等接口
        self._robot = self._Robot(prim_path="/World/Franka", name="franka")
        self.world.scene.add(self._robot)

        # ── ee_link 位置视图（XFormPrimView：标准 Isaac Sim prims API，非底层 USD）──
        # 在 world.reset() 之后才有效，此处只记录路径，初始化在 __init__ 末尾完成
        self._ee_prim_path = "/World/Franka/panda_hand"
        self._ee_view      = None   # 延迟到 world.reset() 后初始化，见 __init__

        # ── 禁用夹爪碰撞体（任务不需要夹爪，避免误触板体）────────────────────────
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdPhysics, UsdGeom
        _stage = get_current_stage()
        for _finger_path in ["/World/Franka/panda_hand","/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"]:
            _prim = _stage.GetPrimAtPath(_finger_path)
            if _prim.IsValid():
                UsdPhysics.CollisionAPI.Apply(_prim).GetCollisionEnabledAttr().Set(False)
                UsdGeom.Imageable(_prim).MakeInvisible()

        # ── 2. 放置带孔板（4 块 FixedCuboid 拼框架）─────────────────────────────
        # 板位于 x = board_x，在 YZ 平面上展开。
        # 框架 4 件：上、下、左、右，中间留空即为孔洞。
        # 孔洞位置在每次 reset 时随机化（见 _reset_board）。
        # 先用零尺寸占位，reset 时通过 USD API 更新 scale 和 position。
        #
        # 命名规则：  /World/Board/Top, Bottom, Left, Right
        placeholder_pos  = np.array([cfg.board_x, 0.0, 0.55]) # 初始化初始位置，这里没有确定4条边框的尺寸，后续 reset 时更新 YZ 坐标
        placeholder_size = np.array([cfg.board_thickness, 0.01, 0.01])  # 暂时极小



        for name in ["Top", "Bottom", "Left", "Right"]:
            piece = self._FixedCuboid(
                prim_path=f"/World/Board/{name}",
                name=f"board_{name.lower()}",       # 唯一 name，避免 scene.add 报重名错误
                position=placeholder_pos.copy(),
                scale=placeholder_size.copy(),
                color=np.array([0.6, 0.6, 0.6]),   # 灰色板件
            )
            self.world.scene.add(piece)
            self._board_pieces.append(piece)

        # ── 3. 目标点可视化（无碰撞，仅显示用）──────────────────────────────
        self._target_marker = self._VisualSphere(
            prim_path="/World/Target",
            radius=0.03,
            color=np.array([1.0, 0.5, 0.0]),   # 橙色目标球
        )
        self.world.scene.add(self._target_marker)

    # ──────────────────────────────────────────────────────────────────────────
    # 配置深度相机（只在 __init__ 中调用一次）
    # ──────────────────────────────────────────────────────────────────────────
    def _setup_camera(self):
        """
        双相机系统：
          1. Wrist Camera  —— 挂载在 ee_link，随 EE 自动运动，clip [0.01, 2.0m]
          2. Global Camera —— 固定俯视整个工作区，clip [0.01, 5.0m]

        深度标注器 (distance_to_image_plane)：
          - 每像素到像平面的正交距离（米），形状 (H, W) float32
          - 无效/无穷远像素为 inf 或 0，由 _normalize_depth() 统一处理
        """
        rep = self._rep
        cfg = self.cfg

        # 关闭自动采集，改为 step()/reset() 里手动触发
        rep.orchestrator.set_capture_on_play(False)

        # ── 1. Global Camera（固定场景坐标系）────────────────────────────────
        self._global_camera = rep.create.camera(
            position=cfg.global_cam_pos,
            look_at=cfg.global_cam_look,
            focal_length=18.0,          # 宽视野，俯瞰全工作区
            horizontal_aperture=24.0,
            clipping_range=(0.01, GLOBAL_DEPTH_MAX),
            name="GlobalCamera",
        )
        global_rp = rep.create.render_product(
            self._global_camera, (IMG_W, IMG_H), name="GlobalView", force_new=True
        )
        self._global_depth_ann = rep.annotators.get("distance_to_image_plane")
        self._global_depth_ann.attach([global_rp])

        # ── 2. Wrist Camera（挂载在 ee_link，随 EE 自动运动）─────────────────
        # rep.create.camera() 不支持 prim_path，改用 USD API 直接在 ee_link
        # 下定义 UsdGeom.Camera prim，作为子节点自动继承 EE 世界变换。
        wrist_cam_path = "/World/Franka/panda_link7/WristCamera"
        stage = self._omni_usd.get_context().get_stage()
        usd_cam = self._UsdGeom.Camera.Define(stage, wrist_cam_path)
        usd_cam.GetFocalLengthAttr().Set(8.0)
        usd_cam.GetHorizontalApertureAttr().Set(24.0)
        usd_cam.GetClippingRangeAttr().Set(self._Gf.Vec2f(0.01, WRIST_DEPTH_MAX))

        # 设置本地偏移和朝向（相对 ee_link 坐标系）
        # USD 相机默认沿本地 -Z 看；180° 绕 X 轴使其对齐 EE Z+ 前向
        xformable = self._UsdGeom.Xformable(usd_cam.GetPrim())
        xformable.ClearXformOpOrder()
        ox, oy, oz = cfg.wrist_cam_offset
        xformable.AddTranslateOp().Set(self._Gf.Vec3d(ox, oy, oz))
        xformable.AddRotateXYZOp().Set(self._Gf.Vec3d(180.0, 0.0, 0.0))

        # render_product 接受 prim path 字符串，rep 自动识别 Camera prim
        wrist_rp = rep.create.render_product(wrist_cam_path, (IMG_W, IMG_H), name="WristView", force_new=True)
        self._wrist_depth_ann = rep.annotators.get("distance_to_image_plane")
        self._wrist_depth_ann.attach([wrist_rp])

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口：reset
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        重置环境到新的 episode 起点。
        每次 reset 随机化：孔洞位置、目标点位置、机械臂初始关节角（小抖动）。

        Returns
        -------
        scalar_obs : (SCALAR_DIM,) float32   标量观测
        depth_img  : (IMG_H, IMG_W) float32  深度图（米）
        info       : dict
        """
        cfg = self.cfg

        # ── 若存在课程切换请求，则在 reset 开始时生效 ───────────────────────
        if self._pending_stage is not None:
            if self._pending_stage != self.cfg.stage:
                self.set_stage(self._pending_stage)
            self._pending_stage = None

        # ── 重置计数器和状态标志 ──────────────────────────────────────────────
        self._step_count   = 0
        self._passed_board = False
        self._prev_action  = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_count += 1

        # ── 随机化孔洞和目标 ──────────────────────────────────────────────────
        hole_y = float(np.random.uniform(*cfg.hole_y_range))  #*代表解包元组，传入两个参数
        hole_z = float(np.random.uniform(*cfg.hole_z_range))
        # 孔心 X 坐标固定在板面处
        self._hole_center  = np.array([cfg.board_x, hole_y, hole_z], dtype=np.float32)

        target_x = float(np.random.uniform(*cfg.target_x_range))
        # 目标 Y/Z 在孔心附近随机（让任务具有连贯性）
        target_y = hole_y + float(np.random.uniform(*cfg.target_y_range))
        target_z = hole_z + float(np.random.uniform(*cfg.target_z_range))
        self._target_pos   = np.array([target_x, target_y, target_z], dtype=np.float32)

        # ── 更新板件几何 ──────────────────────────────────────────────────────
        self._reset_board(hole_y, hole_z)

        # ── 更新目标标记球位置 ────────────────────────────────────────────────
        self._target_marker.set_world_pose(
            position=self._target_pos,   # (3,)
        )

        # ── 重置机械臂关节到初始位置（加小量随机扰动增加多样性）────────────
        init_q = np.array(cfg.init_joints, dtype=np.float32)
        noise  = np.zeros(N_JOINTS_TOTAL, dtype=np.float32)
        _noise_scale = {1: 0.05, 2: 0.02, 3: 0.05, 4: 0.08}.get(cfg.stage, 0.05)
        noise[ACTIVE_JOINT_IDX] = np.random.uniform(-_noise_scale, _noise_scale, len(ACTIVE_JOINT_IDX))
        init_q_noisy = init_q + noise

        # 必须同时做两件事，否则 world.step() 时 PhysX PD 驱动器会把关节拉回默认位置：
        #   1. set_joint_positions  : 瞬移物理状态（关节角、速度）
        #   2. apply_action         : 设置 PD 驱动目标，让物理引擎保持在此位置
        self._robot.set_joint_positions(init_q_noisy)
        self._robot.set_joint_velocities(np.zeros(N_JOINTS_TOTAL, dtype=np.float32))
        self._robot.apply_action(self._ArtAction(joint_positions=init_q_noisy))

        # ── 推进几步让关节完全稳定 ───────────────────────────────────────────
        # 一步不够：PD 控制器需要几帧才能收敛到目标位置
        for _ in range(5):
            self.world.step(render=False)
        self.world.step(render=True)   # 最后一步触发渲染，annotator 更新深度图

        # ── 记录初始 EE 位置 ──────────────────────────────────────────────────
        ee_pos, _ = self._get_ee_pose()
        self._prev_ee_x   = float(ee_pos[0])
        self._prev_ee_pos = ee_pos.copy()   # 用于 step() 中 EE 速度差分估计


        # ── 重置奖励管理器（初始化势能基准）────────────────────────────────
        self._reward_manager.reset(
            ee_pos      = ee_pos,
            hole_center = self._hole_center,
            target_pos  = self._target_pos,
        )

        if cfg.debug_diagnostics and self._episode_count <= cfg.debug_log_episodes:
            collided = self._check_collision(ee_pos)
            done, term_info = self._term_manager.check(
                step_count=0,
                ee_pos=ee_pos,
                passed_board=self._passed_board,
                target_pos=self._target_pos,
                collided=collided,
            )
            print(
                "[debug][reset] ep=%d ee_pos=%s init_q=%s collided=%s done=%s term=%s"
                % (
                    self._episode_count,
                    np.round(ee_pos, 3),
                    np.round(init_q_noisy, 3),
                    collided,
                    done,
                    term_info,
                )
            )

        # ── 获取初始观测 ──────────────────────────────────────────────────────
        scalar_obs   = self.scalar_norm(self._get_scalar_obs(), update=self.training)
        priv_obs     = self.priv_norm(self._get_critic_obs(), update=self.training)
        wrist_depth  = self._get_wrist_depth()
        global_depth = self._get_global_depth()

        info = {
            "hole_center": self._hole_center.copy(),
            "target_pos":  self._target_pos.copy(),
        }

        return scalar_obs, priv_obs, wrist_depth, global_depth, info

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口：step
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        执行一步动作。

        Parameters
        ----------
        action : (ACTION_DIM,) float32，值域 [-1, 1]
                 由 ppo.py / network.py 输出后直接传入，无需外部缩放。

        Returns
        -------
        scalar_obs : (SCALAR_DIM,) float32
        depth_img  : (IMG_H, IMG_W) float32
        reward     : float
        done       : bool
        info       : dict
        """
        cfg = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── 计算关节目标（当前位置 + 动作增量）──────────────────────────────
        current_q = self._robot.get_joint_positions()          # (6,)
        # 6-DOF：action 对应关节 0-5，joint7（index 6）保持不变
        delta_q = np.zeros(N_JOINTS_TOTAL, dtype=np.float32)
        delta_q[ACTIVE_JOINT_IDX] = action * cfg.action_scale  # (5,) → index 0-4
        target_q = current_q + delta_q                          # (6,)

        # ── 发送关节位置指令 ──────────────────────────────────────────────────
        self._robot.apply_action(
            self._ArtAction(joint_positions=target_q)
        )

        # ── 推进物理一步 ──────────────────────────────────────────────────────
        self.world.step(render=True)
        self._step_count += 1

        # ── 获取新状态（PhysX tensor API，无缓存问题）────────────────────────
        ee_pos, ee_quat = self._get_ee_pose()

        # ── EE 速度（差分估计，R_align 和特权观测均需用到）──────────────────
        ee_vel = (ee_pos - self._prev_ee_pos) / cfg.physics_dt   # (3,) m/s

        # ── 碰撞 / 穿孔检测 ───────────────────────────────────────────────────
        collided     = self._check_collision(ee_pos)
        just_passed  = self._check_pass_through(ee_pos)
        # 穿孔后退回板前方惩罚（一次性，防止机器人穿过再退出的策略）
        just_retreated = (self._passed_board and
                          self._prev_ee_x >= self.cfg.board_x and
                          ee_pos[0] < self.cfg.board_x)
        if just_retreated:
            reward_retreat = -3.0
        else:
            reward_retreat = 0.0

        # ── 计算奖励 ──────────────────────────────────────────────────────────
        reward = self._reward_manager.compute(
            ee_pos       = ee_pos,
            ee_vel       = ee_vel,
            action       = action,
            prev_action  = self._prev_action,
            hole_center  = self._hole_center,
            target_pos   = self._target_pos,
            passed_board = self._passed_board,
            collided     = collided,
            just_passed  = just_passed,
        ) + reward_retreat

        # ── 判断 done（委托给 TerminationManager）────────────────────────────
        done, term_info = self._term_manager.check(
            step_count   = self._step_count,
            ee_pos       = ee_pos,
            passed_board = self._passed_board,
            target_pos   = self._target_pos,
            collided     = collided,
        )

        # ── 获取观测 ──────────────────────────────────────────────────────────
        scalar_obs   = self.scalar_norm(self._get_scalar_obs(), update=self.training)
        priv_obs     = self.priv_norm(self._get_critic_obs(), update=self.training)
        wrist_depth  = self._get_wrist_depth()
        global_depth = self._get_global_depth()

        info = {**term_info, "ee_pos": ee_pos.copy(), "stage": self.cfg.stage}

        # ── 课程自动切换：基于滚动 curriculum_window 个回合统计 ──────────────────────────────
        if done and self.training:  # eval episode 不计入课程历史，避免触发错误晋级
            pass_success = 1.0 if self._passed_board else 0.0
            arrive_success = 1.0 if bool(term_info.get("success", False)) else 0.0
            self._pass_history.append(pass_success)
            self._arrive_history.append(arrive_success)

            if len(self._pass_history) == self._curriculum_window:
                pass_rate_100 = float(np.mean(self._pass_history))
                arrive_rate_100 = float(np.mean(self._arrive_history))
                info["pass_rate_100ep"] = pass_rate_100
                info["arrive_rate_100ep"] = arrive_rate_100

                # stage1 -> stage2：穿孔成功率 >= 60%，快速进入过渡阶段
                if self.cfg.stage == 1 and pass_rate_100 >= 0.6:
                    if self._pending_stage is None:
                        self.set_curriculum_stage(2)

                # stage2 -> stage3：穿孔稳定 + 能到达宽松目标
                elif self.cfg.stage == 2 and pass_rate_100 >= 0.8 and arrive_rate_100 >= 0.7:
                    if self._pending_stage is None:
                        self.set_curriculum_stage(3)

                # stage3 -> stage4：穿孔+到达均高度稳定（原 stage2→3 条件）
                elif self.cfg.stage == 3 and pass_rate_100 >= 0.95 and arrive_rate_100 >= 0.8:
                    if self._pending_stage is None:
                        self.set_curriculum_stage(4)

        if cfg.debug_diagnostics and self._episode_count <= cfg.debug_log_episodes:
            if self._step_count <= cfg.debug_log_steps or done:
                bd = self._reward_manager.last_breakdown
                q_after = self._robot.get_joint_positions()
                q_move = float(np.mean(np.abs(q_after[ACTIVE_JOINT_IDX] - current_q[ACTIVE_JOINT_IDX])))
                q_err = float(np.mean(np.abs(target_q[ACTIVE_JOINT_IDX] - q_after[ACTIVE_JOINT_IDX])))
                # 必须在 _prev_ee_pos 更新前计算，否则差值恒为 0
                ee_delta = float(np.linalg.norm(ee_pos - self._prev_ee_pos))
                print(
                    "[debug][step] ep=%d step=%d a_mean=%.4f a_abs=%.4f dq_abs=%.4f q_move=%.4f q_err=%.4f ee_delta=%.4f reward=%.4f done=%s collided=%s term=%s bd=%s"
                    % (
                        self._episode_count,
                        self._step_count,
                        float(action.mean()),
                        float(np.mean(np.abs(action))),
                        float(np.mean(np.abs(delta_q[ACTIVE_JOINT_IDX]))),
                        q_move,
                        q_err,
                        ee_delta,
                        float(reward),
                        done,
                        collided,
                        term_info,
                        {k: round(float(v), 4) for k, v in bd.items()},
                    )
                )

        # ── 更新状态缓存（必须在 debug 块之后，保证 ee_delta 计算正确）─────────
        self._prev_action = action.copy()
        self._prev_ee_x   = float(ee_pos[0])
        self._prev_ee_pos = ee_pos.copy()

        return scalar_obs, priv_obs, wrist_depth, global_depth, reward, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口：close
    # ──────────────────────────────────────────────────────────────────────────
    def close(self):
        """
        关闭 Isaac Sim 仿真。train.py 在训练结束后调用。
        必须先停止 replicator，再关闭 app，否则 omni.graph.core 在 atexit 里崩溃。
        """
        try:
            self._rep.orchestrator.stop()
        except Exception:
            pass
        self._sim_app.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：随机化板件几何
    # ──────────────────────────────────────────────────────────────────────────
    def _reset_board(self, hole_y: float, hole_z: float):
        """
        每个 episode 重新计算 4 块板件（上/下/左/右）的位置和尺寸，
        拼出随机位置的方形孔洞。

        板件命名与位置：

            ┌────────── [Top] ──────────┐   z_high
            │                           │
          [Left]      [HOLE]         [Right]
            │                           │
            └─────── [Bottom] ──────────┘   z_low
           y_min   hole_y-h  hole_y+h  y_max

        其中 h = cfg.hole_half_size

        Parameters
        ----------
        hole_y : 孔中心 Y 坐标（板面坐标系）
        hole_z : 孔中心 Z 坐标（板面坐标系）
        """
        cfg = self.cfg
        bx  = cfg.board_x       # 板所在 X 位置
        th  = cfg.board_thickness
        h   = cfg.hole_half_size
        y0  = -cfg.board_half_y  # 板左沿 Y
        y1  =  cfg.board_half_y  # 板右沿 Y
        z0  = cfg.board_z_low    # 板下沿 Z
        z1  = cfg.board_z_high   # 板上沿 Z

        # 孔洞边界（Clip 防止孔越出板边缘）
        hy_lo = np.clip(hole_y - h, y0 + 0.01, y1 - 0.01)
        hy_hi = np.clip(hole_y + h, y0 + 0.01, y1 - 0.01)
        hz_lo = np.clip(hole_z - h, z0 + 0.01, z1 - 0.01)
        hz_hi = np.clip(hole_z + h, z0 + 0.01, z1 - 0.01)

        # 各板件的 (center_y, center_z, scale_y, scale_z)
        # scale 即 FixedCuboid 的边长（非半边长）
        pieces_geom = {
            "Bottom": (
                (y0 + y1) / 2,           (z0 + hz_lo) / 2,       # center y, z
                y1 - y0,                  hz_lo - z0,             # size y, z
            ),
            "Top": (
                (y0 + y1) / 2,           (hz_hi + z1) / 2,
                y1 - y0,                  z1 - hz_hi,
            ),
            "Left": (
                (y0 + hy_lo) / 2,        (hz_lo + hz_hi) / 2,
                hy_lo - y0,               hz_hi - hz_lo,
            ),
            "Right": (
                (hy_hi + y1) / 2,        (hz_lo + hz_hi) / 2,
                y1 - hy_hi,               hz_hi - hz_lo,
            ),
        }

        names = ["Top", "Bottom", "Left", "Right"]
        for piece, name in zip(self._board_pieces, names):
            cy, cz, sy, sz = pieces_geom[name]
            # 板件厚度方向 = X，Y/Z 方向由孔位决定
            # FixedCuboid 是单体对象，用单数形式 set_world_pose / set_local_scale
            piece.set_world_pose(
                position=np.array([bx, cy, cz])   # (3,)
            )
            piece.set_local_scale(
                np.array([th, max(sy, 0.01), max(sz, 0.01)])  # (3,)
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：获取末端执行器位姿
    # ──────────────────────────────────────────────────────────────────────────
    def _get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取 UR10e 末端执行器（ee_link）的世界坐标系位置和四元数。

        Returns
        -------
        ee_pos  : (3,)  float32  [x, y, z]（米）
        ee_quat : (4,)  float32  [w, x, y, z]（Isaac Sim 约定）

        注意：UR10e 的 EE link 名称为 "ee_link"（来自 Isaac Lab UR10 配置确认）。
        """
        # 懒初始化：create_articulation_view 必须在 world.step() 之后调用
        if self._art_physx_view is None:
            self._init_physx_art_view()
        # omni.physics.tensors 原生 API（IsaacLab 同款）：
        #   update_articulations_kinematic() → PhysX FK：关节角 → link 世界位姿
        #   get_link_transforms()           → (num_arts, num_links, 7) [x,y,z, qx,qy,qz,qw]
        self._physics_sim_view.update_articulations_kinematic()
        t = self._art_physx_view.get_link_transforms()[0, self._ee_link_physx_idx]  # (7,)
        ee_pos  = np.array([t[0], t[1], t[2]], dtype=np.float32)
        # PhysX 四元数格式: [qx, qy, qz, qw] → Isaac Sim 约定: [w, x, y, z]
        ee_quat = np.array([t[6], t[3], t[4], t[5]], dtype=np.float32)
        return ee_pos, ee_quat

    def _init_physx_art_view(self):
        """
        懒初始化 PhysX tensor view，复用 robot 内部已创建的 view（world.step() 后有效）。
        link 索引从 PhysX 实际 link 名称中确定（ee_link 若被 fixed-joint 合并则用 wrist_3_link 代替）。
        """
        self._art_physx_view   = self._robot._articulation_view._physics_view
        self._physics_sim_view = self._SimulationManager.get_physics_sim_view()

        # 从 PhysX 元数据获取真实 link 名称（可能比 URDF 少，因 fixed joint 会合并 link）
        physx_link_names = list(self._art_physx_view.shared_metatype.link_names)
        print(f"[HoleBoardEnv] PhysX links ({len(physx_link_names)}): {physx_link_names}")

        # EE link 索引：优先找 panda_hand，退回 panda_link8
        if "panda_hand" in physx_link_names:
            self._ee_link_physx_idx = physx_link_names.index("panda_hand")
        elif "panda_link8" in physx_link_names:
            self._ee_link_physx_idx = physx_link_names.index("panda_link8")
            print("[HoleBoardEnv] 使用 panda_link8 作为 EE 代理")
        else:
            raise RuntimeError(f"找不到 Franka EE link，实际 links: {physx_link_names}")

        # 特权观测用 arm link 索引（6 条连杆，panda_link7 不控制故不追踪）
        _arm_link_names = [
            "panda_link1", "panda_link2", "panda_link3",
            "panda_link4", "panda_link5", "panda_link6"
        ]
        self._arm_link_physx_idxs = [
            physx_link_names.index(n) for n in _arm_link_names if n in physx_link_names
        ]
        print(
            f"[HoleBoardEnv] EE idx={self._ee_link_physx_idx}, "
            f"arm_idxs={self._arm_link_physx_idxs}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：构造标量观测向量
    # ──────────────────────────────────────────────────────────────────────────
    def _get_scalar_obs(self) -> np.ndarray:
        """
        构造 Actor 标量观测，共 SCALAR_DIM = 31 维：

        [0:6]   sin 编码的 6 个关节角（Franka panda_joint1-6）
        [6:12]  cos 编码的 6 个关节角
                - 将角度编码为 (sin, cos) 避免 ±π 的不连续性
        [12:18] 6 个关节角速度（rad/s）
        [18:21] EE 位置 (x, y, z)（米）
        [21:25] EE 四元数 (w, x, y, z)
        [25:28] EE → 目标点 单位向量 (3D)   ← 真实部署可计算，无特权信息
        [28]    EE → 目标点 欧几里得距离（米）
        [29]    穿孔标志 passed_board ∈ {0, 1}  ← 显式阶段信号
        [30]    存活时间步比例 step_count/max_steps ∈ [0,1]

        注意：Actor 不含 EE→孔洞 信息，必须通过 CNN 感知孔洞位置。

        Returns
        -------
        obs : (31,) float32
        """
        # ── 关节状态（ACTIVE_JOINT_IDX = [0,1,2,3,4,5,6]）───────────────────
        q    = self._robot.get_joint_positions().astype(np.float32)   # (7,)
        qdot = self._robot.get_joint_velocities().astype(np.float32)  # (7,)
        q_act    = q[ACTIVE_JOINT_IDX]     # (7,)
        qdot_act = qdot[ACTIVE_JOINT_IDX]  # (7,)

        sin_q = np.sin(q_act)   # (7,)
        cos_q = np.cos(q_act)   # (7,)

        # ── EE 位姿 ───────────────────────────────────────────────────────────
        ee_pos, ee_quat = self._get_ee_pose()   # (3,), (4,)

        # ── EE → 目标点 ───────────────────────────────────────────────────────
        vec_to_target  = self._target_pos - ee_pos
        dist_to_target = float(np.linalg.norm(vec_to_target))
        unit_to_target = vec_to_target / (dist_to_target + 1e-8)   # (3,) 单位向量

        # ── 存活时间步比例 ─────────────────────────────────────────────────────
        alive_t = np.array([self._step_count / self.cfg.max_steps], dtype=np.float32)

        # ── 拼接原始值（归一化由 self.scalar_norm 统一处理）──────────────────
        obs = np.concatenate([
            sin_q,              # [0:7]   7 dim
            cos_q,              # [7:14]  7 dim
            qdot_act,           # [14:21] 7 dim  rad/s
            ee_pos,             # [21:24] 3 dim  m
            ee_quat,            # [24:28] 4 dim  单位四元数
            unit_to_target,     # [28:31] 3 dim  单位向量
            [dist_to_target],   # [31]    1 dim  m
            [float(self._passed_board)],  # [32]  1 dim  {0,1}
            alive_t,            # [33]    1 dim  ∈ [0,1]
        ], dtype=np.float32)    # total = 34

        assert obs.shape == (SCALAR_DIM,), f"obs shape mismatch: {obs.shape}"
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：构造 Critic 全状态观测向量（纯标量，不含 CNN）
    # ──────────────────────────────────────────────────────────────────────────
    def _get_critic_obs(self) -> np.ndarray:
        """
        构造 Critic 纯标量特权观测，共 PRIVILEGED_DIM = 48 维（全部从仿真直接读取）：

        [0:6]   sin 编码的 6 个关节角
        [6:12]  cos 编码的 6 个关节角
        [12:18] 6 个关节角速度（rad/s）
        [18:21] EE 位置 (x, y, z)
        [21:25] EE 四元数 (w, x, y, z)
        [25:28] 孔洞中心世界坐标 (x, y, z)
        [28:31] 孔洞法向量，固定为 [1, 0, 0]
        [31:34] EE → 孔洞 原始向量（非单位化）
        [34:37] EE → 目标点 原始向量（非单位化）
        [37:43] 6 个活动关节末端到板体的有符号距离（米）
        [43]    EE 工具轴线与法向量夹角 cos(θ)，由 EE 四元数计算
        [44:47] 板中心世界坐标
        [47]    孔洞直径（2 × hole_half_size，反映课程阶段）

        Returns
        -------
        critic_obs : (PRIVILEGED_DIM,) float32
        """
        cfg = self.cfg

        # ── 关节状态（与 Actor 相同编码）────────────────────────────────────
        q    = self._robot.get_joint_positions().astype(np.float32)
        qdot = self._robot.get_joint_velocities().astype(np.float32)
        q_act    = q[ACTIVE_JOINT_IDX]
        qdot_act = qdot[ACTIVE_JOINT_IDX]
        sin_q = np.sin(q_act)
        cos_q = np.cos(q_act)

        # ── EE 位姿 ───────────────────────────────────────────────────────────
        ee_pos, ee_quat = self._get_ee_pose()   # (3,), (4,) [w,x,y,z]

        # ── 孔洞中心 & 法向量 ──────────────────────────────────────────────────
        hole_center = self._hole_center.copy()                         # (3,)
        hole_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)     # 固定

        # ── EE → 孔洞 & EE → 目标点 原始向量 ─────────────────────────────────
        vec_ee_to_hole   = hole_center - ee_pos                        # (3,)
        vec_ee_to_target = self._target_pos - ee_pos                   # (3,)

        # ── 7 个活动关节到板体的有符号距离 ─────────────────────────────────────
        joint_dists = np.zeros(len(ACTIVE_JOINT_IDX), dtype=np.float32)
        link_transforms = self._art_physx_view.get_link_transforms()  # (1, num_links, 7)
        for i, idx in enumerate(self._arm_link_physx_idxs):
            joint_dists[i] = float(link_transforms[0, idx, 0]) - cfg.board_x

        # ── EE 工具轴线（本地 Z 轴）与孔洞法向量 [1,0,0] 的夹角 cos(θ)──────
        # 通过四元数旋转 local-z=[0,0,1] 到世界系，取 X 分量即 dot([1,0,0])
        w, x, y, z = float(ee_quat[0]), float(ee_quat[1]), float(ee_quat[2]), float(ee_quat[3])
        ee_fwd_x = 2.0 * (x * z + w * y)   # world-X component of EE local-Z
        cos_axis_angle = float(np.clip(ee_fwd_x, -1.0, 1.0))

        # ── 板中心坐标（固定几何，反映工作空间位置）───────────────────────────
        board_center = np.array([
            cfg.board_x,
            0.0,
            (cfg.board_z_low + cfg.board_z_high) / 2.0,
        ], dtype=np.float32)

        # ── 孔洞直径（反映课程阶段：stage1/2→0.30m，stage3→0.16m）──────────
        hole_diam = np.array([2.0 * cfg.hole_half_size], dtype=np.float32)

        # ── 拼接（归一化由 self.priv_norm 统一处理）──────────────────────────
        critic_obs = np.concatenate([
            sin_q,              # [0:7]    7 dim
            cos_q,              # [7:14]   7 dim
            qdot_act,           # [14:21]  7 dim
            ee_pos,             # [21:24]  3 dim
            ee_quat,            # [24:28]  4 dim
            hole_center,        # [28:31]  3 dim
            hole_normal,        # [31:34]  3 dim
            vec_ee_to_hole,     # [34:37]  3 dim
            vec_ee_to_target,   # [37:40]  3 dim
            joint_dists,        # [40:47]  7 dim
            [cos_axis_angle],   # [47]     1 dim
            board_center,       # [48:51]  3 dim
            hole_diam,          # [51]     1 dim
        ], dtype=np.float32)    # total = 52

        assert critic_obs.shape == (PRIVILEGED_DIM,), f"critic_obs shape mismatch: {critic_obs.shape}"
        return critic_obs

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：深度图获取与归一化
    # ──────────────────────────────────────────────────────────────────────────
    def _normalize_depth(self, annotator, max_depth: float, label: str) -> np.ndarray:
        """
        从 replicator annotator 读取原始深度，clip 并归一化到 [0, 1]。

        Parameters
        ----------
        annotator : replicator annotator 对象
        max_depth : clip 上限（米），对应归一化后的 1.0
        label     : 用于 WARN 打印，区分 wrist/global

        Returns
        -------
        depth_norm : (IMG_H, IMG_W) float32，值域 [0, 1]
        """
        try:
            raw = annotator.get_data()
            if raw is None or raw.size == 0:
                return np.zeros((IMG_H, IMG_W), dtype=np.float32)
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            depth = raw.astype(np.float32)
            depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
            depth = np.clip(depth, 0.0, max_depth) / max_depth   # 归一化到 [0, 1]
        except Exception as e:
            print(f"[WARN] {label} 深度图获取失败，返回零图: {e}")
            depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        return depth

    def _get_wrist_depth(self) -> np.ndarray:
        """Wrist camera 深度图，归一化到 [0, 1]，clip 上限 WRIST_DEPTH_MAX。"""
        return self._normalize_depth(self._wrist_depth_ann, WRIST_DEPTH_MAX, "wrist")

    def _get_global_depth(self) -> np.ndarray:
        """Global camera 深度图，归一化到 [0, 1]，clip 上限 GLOBAL_DEPTH_MAX。"""
        return self._normalize_depth(self._global_depth_ann, GLOBAL_DEPTH_MAX, "global")

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口：切换课程阶段
    # ──────────────────────────────────────────────────────────────────────────
    # 每个 stage 对应的场景几何参数覆盖值
    # 不在此表中的 stage 使用 EnvConfig 默认值
    _STAGE_GEOM = {
        # stage 1 沿用 EnvConfig 默认几何
        # stage 2：success_dist 放宽 + 目标紧贴孔洞正后方，穿过直走即可到达
        2: dict(success_dist=0.06,
                target_x_range=(0.35, 0.4),
                target_y_range=(-0.05, 0.05),
                target_z_range=(-0.05, 0.05)),
        # stage 3：桥接 stage2 → stage4，target 偏移量适度扩大，避免斜角穿孔碰 body
        3: dict(success_dist=0.05,
                hole_y_range  = (-0.15, 0.15),
                target_x_range=(0.35, 0.45),
                target_y_range=(-0.1, 0.1),
                target_z_range=(-0.1, 0.1)),
        # stage 4（原 Stage 3）：位置随机化加大，到达判定收紧
        4: dict(
            hole_half_size  = 0.16,
            hole_y_range    = (-0.15,  0.15),
            hole_z_range    = (0.5,  0.60),
            target_x_range=(0.35, 0.5),
            target_y_range  = (-0.1, 0.1),
            target_z_range  = (-0.1, 0.1),
            success_dist    = 0.04,
        ),
    }

    def set_stage(self, stage: int):
        """切换课程阶段（1/2/3/4），同步更新 RewardManager 权重和场景几何配置。"""
        assert stage in (1, 2, 3, 4), f"stage 必须是 1/2/3/4，got {stage}"
        self.cfg.stage = stage
        self._reward_manager.set_stage(stage)

        # ++ 清空滚动窗口，避免上一阶段历史污染新阶段的晋级判断
        self._pass_history.clear()
        self._arrive_history.clear()

        # 应用 per-stage 几何覆盖（恢复默认值后再覆盖，支持 stage 回退）
        self.cfg.hole_half_size  = 0.18
        self.cfg.hole_y_range    = (-0.1, 0.1)
        self.cfg.hole_z_range    = (0.55,  0.58)
        self.cfg.target_y_range  = (-0.1, 0.1)
        self.cfg.target_z_range  = (-0.1, 0.1)
        self.cfg.success_dist    = 0.05
        geom = self._STAGE_GEOM.get(stage, {})
        for k, v in geom.items():
            setattr(self.cfg, k, v)
        # 同步 TerminationManager 的 success_dist
        self._term_manager._success_dist = self.cfg.success_dist

        print(f"[stage] switched to {stage} | hole_half={self.cfg.hole_half_size:.3f}m "
              f"target_y={self.cfg.target_y_range} success_dist={self.cfg.success_dist:.3f}m")

    def set_curriculum_stage(self, stage: int):
        """请求课程切换到指定阶段，切换将在下次 reset 时生效。"""
        assert stage in (1, 2, 3, 4), f"stage 必须是 1/2/3/4，got {stage}"
        if self._pending_stage == stage:
            return
        self._pending_stage = stage
        print(f"[curriculum] queued stage {stage} (apply on next reset)")

    @property
    def reward_breakdown(self) -> dict:
        """返回上一步各奖励 term 的值，供 evaluate_policy 统计用。"""
        return self._reward_manager.last_breakdown

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：碰撞检测（几何法）
    # ──────────────────────────────────────────────────────────────────────────
    def _check_collision(self, ee_pos: np.ndarray) -> bool:
        """
        用几何方法检测 EE 是否与板件碰撞。

        判断逻辑：
          1. EE 的 X 坐标在板的厚度范围内（穿过板平面）
          2. EE 的 Y/Z 坐标落在板的实体区域（不在孔洞内）

        这种方法比物理接触传感器更稳定（不受物理步长影响），
        且足够精确（EE 半径远小于孔洞余量）。

        Parameters
        ----------
        ee_pos : (3,) EE 当前世界坐标

        Returns
        -------
        bool : True = 检测到碰撞
        """
        cfg = self.cfg
        bx  = cfg.board_x
        th  = cfg.board_thickness / 2.0

        # 条件1：EE X 坐标在板厚度范围内
        x_in_board = (bx - th) < ee_pos[0] < (bx + th)
        if not x_in_board:
            return False

        # 条件2：EE Y/Z 在板的整体范围内（否则已超出板外，无碰撞）
        y_in_board = -cfg.board_half_y < ee_pos[1] < cfg.board_half_y
        z_in_board = cfg.board_z_low   < ee_pos[2] < cfg.board_z_high
        if not (y_in_board and z_in_board):
            return False

        # 条件3：EE 在孔洞范围内则无碰撞，否则碰撞
        h  = cfg.hole_half_size
        hy = self._hole_center[1]
        hz = self._hole_center[2]
        in_hole = (abs(ee_pos[1] - hy) < h) and (abs(ee_pos[2] - hz) < h)

        return not in_hole   # 在板体实心区域 = 碰撞

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：穿孔检测
    # ──────────────────────────────────────────────────────────────────────────
    def _check_pass_through(self, ee_pos: np.ndarray) -> bool:
        """
        检测 EE 是否刚完成穿孔（从板前 → 板后的跨越）。

        判断条件：
          - 上一步 EE 在板前 (prev_x < board_x)
          - 当前步 EE 在板后 (curr_x > board_x)
          - 跨越时 EE 的 Y/Z 在孔洞范围内（否则算碰穿，不算穿孔）

        注意：_check_collision 会先于此函数判断，碰撞时 done=True，
        穿孔检测不会在碰撞帧触发。

        Returns
        -------
        bool : True = 刚完成穿孔
        """
        cfg = self.cfg
        bx  = cfg.board_x

        if self._passed_board:
            return False   # 已穿过，不重复计

        just_crossed = self._prev_ee_x < bx <= ee_pos[0]
        if not just_crossed:
            return False

        # 验证穿越时 EE 在孔洞内
        h  = cfg.hole_half_size
        hy = self._hole_center[1]
        hz = self._hole_center[2]
        in_hole = (abs(ee_pos[1] - hy) < h) and (abs(ee_pos[2] - hz) < h)

        if in_hole:
            self._passed_board = True
            return True

        return False

    # ──────────────────────────────────────────────────────────────────────────
    # 属性：供 ppo.py 查询维度，不需要实例化才能获取
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def scalar_obs_dim(self) -> int:
        """标量观测维度，供 network.py 构建网络时使用。"""
        return SCALAR_DIM

    @property
    def depth_img_shape(self) -> Tuple[int, int]:
        """深度图分辨率 (H, W)，供 network.py 构建 CNN 时使用。"""
        return (IMG_H, IMG_W)

    @property
    def action_dim(self) -> int:
        """动作维度，供 network.py / ppo.py 使用。"""
        return ACTION_DIM




# ─────────────────────────────────────────────────────────────────────────────
# 独立测试入口（python env.py 直接运行验证环境搭建）
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("  env.py 独立测试模式")
    print("  验证：场景搭建 / reset / step 接口")
    print("  关闭窗口或 Ctrl+C 退出")
    print("=" * 50)

    # SimulationApp 必须在最外层创建，且必须是脚本第一个语句
    sim_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

    cfg = EnvConfig(headless=False, max_steps=200)
    env = HoleBoardEnv(cfg, sim_app=sim_app)

    try:
        # ── 第一个 episode：打印观测维度 ──────────────────────────────────────
        scalar_obs, priv_obs, wrist_depth, global_depth, info = env.reset()
        print(f"\n[reset] scalar_obs shape  : {scalar_obs.shape}   期望: ({SCALAR_DIM},)")
        print(f"[reset] priv_obs shape    : {priv_obs.shape}   期望: ({PRIVILEGED_DIM},)")
        print(f"[reset] wrist_depth shape : {wrist_depth.shape}   期望: ({IMG_H}, {IMG_W})")
        print(f"[reset] global_depth shape: {global_depth.shape}   期望: ({IMG_H}, {IMG_W})")
        print(f"[reset] hole_center       : {info['hole_center']}")
        print(f"[reset] target_pos        : {info['target_pos']}")
        print("\n窗口已打开，保持静止中... 关闭窗口退出。\n")

        # ── 持续运行，episode 结束后自动 reset，让窗口保持可见 ────────────────
        episode = 0
        while sim_app.is_running():
            action = np.zeros(ACTION_DIM, dtype=np.float32)  # 暂时关闭随机动作，保持静止
            # action = np.random.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
            obs, priv, wrist_d, global_d, reward, done, info = env.step(action)

            if done:
                episode += 1
                result = ("SUCCESS  " if info["success"]
                          else "COLLISION" if info["collision"]
                          else "TIMEOUT  ")
                print(f"[ep {episode:3d}]  {result}  "
                      f"steps={info['step_count']:3d}  "
                      f"dist_to_target={info['dist_to_target']:.3f}m")
                # 重置进入下一个 episode
                obs, priv, wrist_d, global_d, info = env.reset()

    except KeyboardInterrupt:
        print("\n[test] 用户中断。")
    finally:
        # 必须先 stop replicator 再 close，否则 atexit 里 omni.graph.core 崩溃
        env.close()
        sys.exit(0)   # 强制退出，跳过 Python atexit 的剩余清理（Isaac Sim 已自己清理）
