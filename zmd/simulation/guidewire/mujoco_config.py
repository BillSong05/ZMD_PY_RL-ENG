"""MuJoCo 导丝物理参数 dataclass(面板 ↔ backend 唯一桥)。

默认值参考 cathsim/airvlab + Nitinol(镍钛合金)医疗导丝典型数据。

单位约定:
- 长度: 与项目其它模块一致(本项目血管 STL 是 mm 量级,我们也用 mm)
- 密度: kg/m^3
- 刚度/阻尼: MuJoCo joint stiffness/damping 单位
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


@dataclass
class MujocoGuidewireConfig:
    # 几何 — 全部锁在面板调好的实验值,用户不再通过 UI 调整。
    n_segments: int = 78
    """capsule 链段数。MuJoCo XML 嵌套 body 超过 ~78 层会爆栈,78 是安全上限。"""

    segment_length: float = 2.564
    """每段长度(mm)。n_segments(78) × segment_length(2.564) ≈ 200 mm = 20 cm 导丝总长。"""

    radius: float = 0.1
    """导丝半径(mm)。

    本项目 STL 是缩小版血管(min lumen radius ≈ 0.265 mm),
    所以默认用 0.1 mm(约 0.008")—— 在最细分支里仍有 ~62% lumen 余量。
    用户可通过面板调节,或调 ``MujocoGuidewire.auto_radius_from_centerline``。
    临床真实 0.014" 导丝半径约 0.18 mm,适用于足尺血管模型。
    """

    # 刚度 / 阻尼(cathsim 的近段刚 → 远段软 cascade)
    stiffness: float = 25.0
    """近端段的初始 hinge 刚度。"""

    stiffness_decay: float = 0.995
    """每往远端推进一段,刚度乘以这个系数(cathsim 默认 0.995)。"""

    damping: float = 1.5
    """每段 hinge 阻尼。"""

    proximal_slide_damping: float = 2.0
    """近端 slide 关节阻尼(送丝阻力)。"""

    proximal_twist_damping: float = 1.0
    """近端 twist 关节阻尼。"""

    armature: float = 0.02
    """关节转子惯量(数值稳定性,cathsim 用 0.05)。"""

    # 物理材料
    density: float = 6720.0
    """Nitinol 密度 6720 kg/m^3(cathsim 用 672 那是 g/cm^3 误植,医学文献 6.45-6.72 g/cc)。"""

    friction_slide: float = 0.1
    """滑动摩擦系数(condim=1 时只用这个)。"""

    contact_condim: int = 3
    """接触条件维数:1=滑动摩擦,3=+扭转,4=+滚动,6=全。导丝-血管推荐 3。"""

    contact_margin: float = 0.05
    """接触检测 margin(mm)。"""

    # J-tip 预弯 — 同时影响物理转向权重(sin(angle))和可视弧长。
    # 锁定为面板调好的实验值 65°(用户最终选择,UI 已删,不再可调)。
    j_tip_angle_deg: float = 65.0
    """末端预弯角度(度),0 表示直丝。锁定 65°(实验调好的值)。

    这个参数同时驱动两件事:
    1. 物理转向:_j_tip_effective_weight = steering_weight × sin(angle)。
    2. 视觉 J-tip overlay 弧角(配合 marker 显示 twist 旋转)。"""

    j_tip_segments: int = 4
    """末端预弯涉及的段数(从最末端起算),配合 segment_length=1.5 给 6mm 可视弧长。

    backend XML 构造时用此分摊 J 弯曲弧度;kinematic 模式不影响 raw 计算。"""

    # J-tip steering(让 twist 旋转真正改变前进方向)
    j_tip_steering_enabled: bool = True
    """是否启用 J-tip 转向:twist 旋转 → J-tip 偏转方向 → move_dir。"""

    j_tip_steering_weight: float = 0.35
    """J-tip 偏转方向对 move_dir 的权重(建议 0.25~0.45)。
    太小旋转没效果,太大导丝乱偏。最终权重还会乘 sin(j_tip_angle_deg)。"""

    j_tip_min_angle_deg: float = 1.0
    """J-tip 角度小于此值(如 0°=直丝)时不启用转向。"""

    # 碰撞
    enable_collision: bool = False
    """是否加载血管 hull 集做 MuJoCo mesh 碰撞。

    当前(2026-05)默认关闭 — 实测真实血管 STL 凸分解后碰撞质量差,
    导丝会卡死或穿模。改用 CenterlineTubeConstraint 软约束在 backend 内统一处理。
    保留这个开关供后续真实碰撞方案恢复(VHACD/CoACD 装上时)。
    """

    enable_centerline_constraint: bool = True
    """是否对每个 body 施加"不离开管腔"的弹簧软约束(沿中心线)。"""

    constraint_k_spring: float = 5000.0
    """约束弹簧刚度(径向 N/mm)。"""

    constraint_k_damp: float = 20.0
    """约束径向阻尼。"""

    constraint_margin: float = 0.02
    """允许的径向余量(mm):允许 = lumen - guidewire_radius - margin,超过开始施力。"""

    constraint_k_attract: float = 0.0
    """额外的"中心吸引力"刚度:>0 时所有 body 都被轻拉向最近中心线点,
    用于让导丝主动跟随弯曲中心线(不仅是不越界)。
    """

    # 弱中心线方向引导 — 中心线只作"判断参考",不做"自动牵引"
    # ----------------------------------------------------------------
    # 介入实验真实感:导丝姿态(物理头方向 + J-tip 旋转)由操作者决定,
    # 中心线只用来判定"当前方向是否大致正确";方向错了系统不会自动救,
    # 必须靠操作者手动旋转 J-tip 拉回来。
    centerline_bias: float = 0.15
    """对齐时中心线方向引导权重(只在 angle_to_centerline < align_angle_deg 时生效)。

    aligned 分支:move_dir = (1-bias)*raw_move_dir + bias*center_tangent。
    bias=0.0 → 完全用 raw(更"硬核",刚出发就可能贴壁);
    bias=0.15 → 轻微参考中心线,避免出发即贴壁,但不构成牵引;
    bias 越大,越像"自动导航",违背介入手感。

    重要:偏航时(angle >= align_angle_deg)忽略此权重,纯走 raw_move_dir,
    让错误方向自然产生顶壁/卡住,操作者必须手动转回来。
    """

    align_angle_deg: float = 30.0
    """"对齐"阈值 — 仅决定是否在 move_dir 里混入 center_tangent(centerline_bias)。

    angle_to_centerline < align_angle_deg:
      → aligned,status=centerline_assist,move_dir = (1-bias)*raw + bias*center_tangent。
    angle_to_centerline >= align_angle_deg:
      → 偏航,status=raw_forward,move_dir = raw_move_dir(不混入中心线)。

    注意:这个阈值**不再单独决定 wall-slide 是否允许**(那会让导丝一碰壁就停)。
    wall-slide 由下面两个更宽松的 knob 联合判定。
    """

    slide_angle_max: float = 55.0
    """允许 wall-slide 的"中心线偏角"上限(度)。

    angle_to_centerline < slide_angle_max → 直接允许 wall-slide(reason="aligned" 或
    "yaw_tolerable")。覆盖大部分"还在朝血管方向走、只是轻微偏航"的情况。

    > align_angle_deg 但仍 < slide_angle_max 的区间是关键 — 旧逻辑这里直接 blocked,
    新逻辑允许沿壁滑行,避免"碰壁就停"。
    """

    yaw_slide_thresh: float = 70.0
    """允许 wall-slide 的"投影偏离"上限(度)。

    candidate_slide_dir = project(raw_move_dir, wall_normal);
    projection_angle = angle(candidate_slide_dir, raw_move_dir):
    - < yaw_slide_thresh → 即使 angle_to_centerline >= slide_angle_max 也允许滑行
      (因为投影方向仍大致沿操作者意图,壁面相当于一个"软栅栏"导丝);
    - >= yaw_slide_thresh → 投影方向已严重偏离 raw,禁止 wall-slide,必须靠
      操作者手动旋转 J-tip 重新对齐。

    几何含义:wall_normal ⊥ raw_move_dir 时投影角 ≈ 0°(墙是软栅栏,完全允许);
    wall_normal // raw_move_dir 时投影角 → 90°(撞死墙,无法借滑动前进)。
    """

    max_centerline_deviation: float = 0.0
    """[已弃用 — 中心线只作判断参考,不再约束偏离距离]

    旧逻辑在偏离 > 此阈值时强制把方向切到 center_tangent → 自动纠偏,
    违背"中心线不是牵引"的原则。保留字段供旧测试兼容,backend 不再读取。
    """

    collision_margin: float = 0.04
    """血管 SDF 碰撞安全余量(mm)。

    导丝不穿模的判定条件:sd <= -(guidewire_radius + collision_margin)
    其中 sd 由 vtkImplicitPolyDataDistance 给出,负值=腔内,正值=腔外。
    不满足时先尝试切向滑动(enable_wall_slide=True),滑动也失败才 blocked。
    0.0 时安全判定退化为 sd <= -guidewire_radius(Minkowski 膨胀边界)。
    """

    # 碰壁切向滑动
    enable_wall_slide: bool = True
    """碰壁后是否尝试沿壁面切向滑动。

    True  — candidate 不安全时,先估计 SDF 法向,把 move_dir 投影到切平面,
            沿切向方向尝试滑动;成功则 status=wall_slide,失败再 blocked。
    False — 遇壁直接 blocked(旧行为)。
    """

    wall_slide_step_scale: float = 0.75
    """切向滑动步长比例(相对原始 step_size)。

    切向基准步长 = step_size * wall_slide_step_scale。
    之后每次尝试再缩半(×0.5^i),共 wall_slide_attempts 次。
    """

    sdf_normal_eps: float = 0.05
    """有限差分估计 SDF 梯度的 eps(mm)。

    太小会被 SDF 数值噪声淹没,太大会模糊曲率细节。
    推荐 0.03~0.08 mm,默认 0.05 mm。
    """

    wall_slide_attempts: int = 3
    """切向滑动最大尝试次数。

    第 i 次(0-based)的步长为 base_step * 0.5^i。
    全部失败后才报 sdf_blocked。
    """

    # 整段导丝 SDF 检查(不只检查 tip,防止前端某段局部穿壁)
    sdf_check_front_points: int = 8
    """SDF 检查导丝前端多少个可见点(导丝是有半径的 tube,只查 tip 会局部穿壁)。"""

    sdf_check_segment_samples: int = 3
    """相邻可见点之间额外插值检查多少个点(避免两端点安全但中间线段穿壁)。"""

    # 小窗"导丝头部第一人称跟随"相机
    # 严格语义:相机绑在 tip 后方 back_offset(沿导丝 heading 反方向),focal 在 tip
    # 前方 forward_distance(沿 heading 正方向)。导丝转向时 heading 变,小窗视角跟着转,
    # 类似内窥镜/术中导丝观察。side/up 偏移已废弃(那是第三人称语义)。
    #
    # 距离按本项目血管尺度选(整体 ~78mm,lumen 直径中位数 ~2.3mm,wire 段长 1.5mm)。
    # 临床真实尺度的导丝 back 1-3mm / lookahead 5-15mm 对这套缩小模型过远 ——
    # 会让小窗看起来像"血管全景放大镜",而不是"贴近 tip 的第一人称"。
    inset_camera_back_offset: float = 1.2
    """相机相对 tip 沿 heading 反方向后退距离(mm)。1.2 ~ 中位 lumen 半径,相机
    在腔内但有距离能把 tip 看进画面;再小会贴穿 wire 的 tube 本身。"""

    inset_camera_side_offset: float = 0.0
    """[已废弃] 不再使用,保持 0。"""

    inset_camera_up_offset: float = 0.6
    """相机相对 tip 沿"视角 up 方向"抬高的距离(mm)。略抬,让 tip 处于画面中下
    1/3,前方腔道在上半部分。0.6 < lumen 半径,相机仍在腔内。"""

    inset_camera_forward_distance: float = 0.0
    """[已废弃] focal 现在锚定在 tip,不再 lookahead。保持字段供旧接口兼容。"""

    inset_camera_smooth: float = 0.2
    """相机位置 / 焦点的低通平滑(0~1)。0.2 = 约 1.25 tick,贴 tip 跟手。"""

    inset_heading_smooth: float = 0.85
    """heading(forward 方向)单独的低通平滑(0~1)。

    分离 heading 平滑是关键 —— _estimate_tip_direction 受 substep 噪声影响,
    每帧 fwd 抖动会同时驱动相机位置和视向晃动,小窗"乱"。把 heading 单独重
    平滑(0.85,~6 tick 时间常数),让"视向"几乎不随 substep 抖,但相机位置
    仍能在 ~1.25 tick 内跟上 tip(由 inset_camera_smooth=0.2 控制)。"""

    hulls_dir: Optional[Path] = None
    """凸分解后的 hull 目录。None 时由 backend 自动找 projects/cache/hulls/。"""

    # 仿真步进
    timestep: float = 0.002
    """MuJoCo 仿真步长(s)。"""

    mj_substeps_per_action: int = 3
    """每次 UI 动作触发的 MuJoCo 步数。"""

    action_substeps: int = 6
    """每次 UI 动作内部分几次推进(每 substep 独立做安全检查/滑动/回滚)。

    6 substep × push_step 0.5mm = 0.083mm/substep,弯道处更细腻地"找可行方向"。
    一个 substep blocked 不影响前面成功的 substep(qpos 只回滚该 substep)。
    """

    wall_slide_inward_bias_factor: float = 0.6
    """wall-slide 候选点的内推安全余量,以"导丝半径"为单位。

    旧默认 collision_margin × 0.5 = 0.02mm,小于 SDF 数值噪声 (~0.01mm 量级),
    导致候选点经常落在边界外侧而被 _is_candidate_wire_safe 拒绝(整步 blocked)。

    新公式:inward_bias = -wall_normal × max(margin × 2, radius × factor)。
    factor=0.6 时,wire_radius=0.1 → 内推 0.06mm,显著大于 SDF 噪声。
    """

    # 启动状态
    initial_inserted_length: float = 0.0
    """初始已插入长度(mm),0 = 完全在体外。"""

    def copy(self, **overrides) -> "MujocoGuidewireConfig":
        return replace(self, **overrides)

    @property
    def max_length(self) -> float:
        return float(self.n_segments) * float(self.segment_length)


def default_config() -> MujocoGuidewireConfig:
    return MujocoGuidewireConfig()
