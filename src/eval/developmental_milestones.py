"""Developmental milestone scale — measuring how close the agent is to the
8–15-year-old North Star via cognitive-science age-graded tasks.

为什么需要这个 (open-gap C#8):
    整个训练路线 (Stage5 → Stage6 → CoreKnowledge → Y1 → MiniGrid → 3D) 的
    exit 标准目前是 "30 天不间断 / 10 任务 / 显存趋平"——这些衡量 *系统韧性*,
    不衡量 *认知能力到达几岁*。没有这把尺,路线是盲飞。本模块提供一把
    项目自定的 "发育里程碑量表",把认知科学年龄分级任务映射成 karbon 可在
    PhysicsSandbox 上自动评测的指标。

设计:
    - 每个 Milestone 有:名称、对应人类年龄、自动评测函数、达标阈值。
    - 评测基于 PhysicsSandbox 的可观测状态 (proprio + 物体列表),不依赖外部。
    - 输出 "估计认知年龄" = 已达标里程碑的最大年龄。

注意:这是 *受局限的* 近似量表 (只在 PhysicSandbox 可控域),不是通用 IQ 测试。
但足以给训练路线一个闭环反馈信号。

有界:评测不创建无界结构;物体数受 env 容量约束。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# =====================================================================
# Milestone 量表 (认知科学年龄分级, 映射成 PhysicSandbox 可测任务)
# =====================================================================


@dataclass
class Milestone:
    key: str
    name: str
    age_years: float          # 对应人类典型达成年龄
    description: str
    # 评测函数签名: (env_state: dict) -> float in [0, 1] (达标度)
    evaluate: Callable[[dict], float]
    threshold: float = 0.6    # 达标度 >= threshold 视为 "已掌握"


# ---------------------------------------------------------------------
# 评测辅助:从 env 暴露的状态里提取物理量
# env_state 约定字段:
#   "agent": (x, y, vx, vy)
#   "objects": list of dict {"x","y","vx","vy","color","tag","static"}
#   "occluded": 被遮挡物体列表 (用于客体永存)
#   "actions": 最近动作序列
# ---------------------------------------------------------------------


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- 里程碑 1: 客体永存 (object permanence) ~ 1 岁 ---
# 评测:物体被遮挡/移出视野后,agent 是否仍朝其最后已知位置搜索
# (用 "遮挡期间 agent 是否保持朝最后位置移动" 近似)
def _eval_object_permanence(st: dict) -> float:
    occ = st.get("occlusion_events", [])
    if not occ:
        return 0.0
    correct = 0
    for ev in occ:
        # ev: {"last_known": (x,y), "agent_traj_during_occ": [(x,y), ...]}
        lk = ev["last_known"]
        traj = ev.get("agent_traj_during_occ", [])
        if not traj:
            continue
        # 遮挡期间 agent 是否朝 last_known 靠近
        start_d = _dist(traj[0], lk)
        end_d = _dist(traj[-1], lk)
        if end_d < start_d * 0.7:
            correct += 1
    return correct / len(occ)


# --- 里程碑 2: 直觉物理 (intuitive physics) ~ 2-3 岁 ---
# 评测:agent 施力方向是否与物体运动方向一致 (力→动因果)
def _eval_intuitive_physics(st: dict) -> float:
    pairs = st.get("force_motion_pairs", [])
    if not pairs:
        return 0.0
    ok = 0
    for p in pairs:
        f = p["force"]          # (fx, fy)
        v = p["velocity_after"]  # (vx, vy)
        fn = math.hypot(*f)
        vn = math.hypot(*v)
        if fn < 1e-6 or vn < 1e-6:
            continue
        cos = (f[0] * v[0] + f[1] * v[1]) / (fn * vn)
        if cos > 0.5:   # 运动方向与施力方向大致一致
            ok += 1
    return ok / len(pairs)


# --- 里程碑 3: 数感 (number sense) ~ 3-4 岁 ---
# 评测:agent 对物体数量的估计误差 < 1 (在小数量范围)
def _eval_number_sense(st: dict) -> float:
    trials = st.get("count_trials", [])
    if not trials:
        return 0.0
    errs = []
    for t in trials:
        true_n = t["true_count"]
        est_n = t["estimated_count"]   # 由 agent 的内部数感头输出
        if true_n == 0:
            continue
        errs.append(abs(est_n - true_n) / true_n)
    if not errs:
        return 0.0
    mean_err = float(np.mean(errs))
    # 误差 0 -> 1.0, 误差 >=0.5 -> 0.0
    return max(0.0, 1.0 - mean_err * 2.0)


# --- 里程碑 4: 手段-目的 (means-ends) ~ 1.5 岁 (占位接口) ---
def _eval_means_ends(st: dict) -> float:
    # TODO: 需 env 提供 "目标是被遮挡物, 需绕路/借助工具" 的任务轨迹
    return st.get("means_ends_score", 0.0)


# --- 里程碑 5: 心智理论 (false-belief) ~ 4 岁 (占位接口) ---
def _eval_theory_of_mind(st: dict) -> float:
    # TODO: 需 3D / 社会教师环境喂信号; PhysicsSandbox 无此信号
    return st.get("tom_score", 0.0)


# --- 里程碑 6: 系统推理 / 守恒 (conservation) ~ 7-11 岁 (占位接口) ---
def _eval_systematic_reasoning(st: dict) -> float:
    # TODO: 需符号/逻辑任务 (Y1 神经符号支线提供)
    return st.get("systematic_score", 0.0)


# 量表 (按年龄升序)
MILESTONES: list[Milestone] = [
    Milestone("object_permanence", "客体永存", 1.0,
              "物体消失后仍相信其存在并搜索", _eval_object_permanence),
    Milestone("means_ends", "手段-目的", 1.5,
              "为达目标使用中介手段", _eval_means_ends),
    Milestone("intuitive_physics", "直觉物理", 2.5,
              "理解施力→运动的因果方向", _eval_intuitive_physics),
    Milestone("number_sense", "数感", 3.5,
              "小数量物体计数误差<1", _eval_number_sense),
    Milestone("theory_of_mind", "心智理论(错误信念)", 4.0,
              "理解他者可有不同信念", _eval_theory_of_mind),
    Milestone("systematic_reasoning", "系统推理/守恒", 9.0,
              "在符号任务上做系统逻辑推演", _eval_systematic_reasoning),
]


# =====================================================================
# 评测器
# =====================================================================


@dataclass
class MilestoneReport:
    scores: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)
    estimated_age: float = 0.0

    def summary(self) -> str:
        lines = ["Developmental Milestone Report:"]
        for m in MILESTONES:
            s = self.scores.get(m.key, 0.0)
            mark = "PASS" if self.passed.get(m.key) else "  - "
            lines.append(f"  [{mark}] {m.age_years:>4}y {m.name:16s} score={s:.2f}")
        lines.append(f"  -> estimated cognitive age ≈ {self.estimated_age:.1f} y")
        return "\n".join(lines)


class DevelopmentalEvaluator:
    """在给定 env_state 序列上评测全部里程碑, 输出发育年龄估计。

    有界: 只读取传入的状态字典, 不创建无界缓冲。
    """

    def evaluate(self, env_states: list[dict]) -> MilestoneReport:
        # 把状态序列聚合成单个聚合状态 (各评测函数自行处理序列语义)
        agg = self._aggregate(env_states)
        report = MilestoneReport()
        max_age = 0.0
        for m in MILESTONES:
            try:
                score = float(m.evaluate(agg))
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))
            passed = score >= m.threshold
            report.scores[m.key] = score
            report.passed[m.key] = passed
            if passed:
                max_age = max(max_age, m.age_years)
        report.estimated_age = max_age
        return report

    @staticmethod
    def _aggregate(states: list[dict]) -> dict:
        """把多步状态聚合成评测函数期望的聚合字典。

        评测函数需要的序列信号 (遮挡事件 / 力-动对 / 计数试次) 由 env
        在 info 里累积提供; 这里做简单合并。
        """
        agg: dict = {}
        # 合并 occlusion / force-motion / count 列表
        for key in ("occlusion_events", "force_motion_pairs", "count_trials"):
            merged = []
            for st in states:
                merged.extend(st.get(key, []))
            agg[key] = merged
        # 透传最新单值分数
        if states:
            last = states[-1]
            for k in ("means_ends_score", "tom_score", "systematic_score"):
                if k in last:
                    agg[k] = last[k]
        return agg


def estimate_cognitive_age(env_states: list[dict]) -> MilestoneReport:
    """便捷函数: 直接给状态序列, 返回报告。"""
    return DevelopmentalEvaluator().evaluate(env_states)
