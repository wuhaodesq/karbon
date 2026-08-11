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
    """Evaluate understanding of physical causality.

    Multi-signal approach:
    1. Force-motion alignment (original): pushing object -> it moves
    2. Grasp-carry physics: held object follows agent (object permanence in motion)
    3. Release-drop physics: released object stops/decelerates (gravity/inertia)
    4. Tool-use physics: held object transfers force to another object
    """
    pairs = st.get("force_motion_pairs", [])
    scores = []

    # Signal 1: Force-motion alignment (original)
    if pairs:
        ok = 0
        for p in pairs:
            f = p["force"]
            v = p["velocity_after"]
            fn = math.hypot(*f)
            vn = math.hypot(*v)
            if fn < 1e-6 or vn < 1e-6:
                continue
            cos = (f[0] * v[0] + f[1] * v[1]) / (fn * vn)
            if cos > 0.2:
                ok += 1
        scores.append(ok / len(pairs))

    # Signal 2: Grasp-carry events (held object moves with agent = understands attachment)
    grasp_events = st.get("grasp_carry_events", [])
    if grasp_events:
        # Each grasp-carry event where object moved with agent counts as physics understanding
        scores.append(min(1.0, len(grasp_events) / 5.0))

    # Signal 3: Tool-use events (agent uses held object to affect another)
    tool_events = st.get("tool_use_events", [])
    if tool_events:
        # Agent understands that held object can transfer force
        scores.append(min(1.0, len(tool_events) / 3.0))

    # Signal 4: Release events (agent understands object detaches and is independent)
    release_events = st.get("release_events", [])
    if release_events:
        scores.append(min(1.0, len(release_events) / 5.0))

    if not scores:
        return 0.0
    return float(np.mean(scores))


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


# --- 里程碑 4: 手段-目的 (means-ends) ~ 1.5 岁 ---
# 评测: agent 是否使用间接手段达成目标。
# 通过 force_motion_pairs 检测 "链式反应":
#   agent → 物体A → 物体B (非直接接触的因果链)
# 同时检测 agent 是否在物体间做有序操作。
def _eval_means_ends(st: dict) -> float:
    """Evaluate indirect/goal-directed action (means-ends reasoning).

    Multi-signal approach:
    1. Explicit env score (chain task completion)
    2. Chain reactions (original): push A -> A hits B
    3. Grasp-carry-release: agent picks up object, carries it, releases at goal
    4. Tool use: agent uses held object to affect another object
    5. Ordered contact: systematic object exploration
    """
    # Check for explicit score first (env can provide direct signal)
    explicit = st.get("means_ends_score")
    if explicit is not None and explicit > 0:
        return float(explicit)

    scores = []

    # Signal 1: Chain reactions (original)
    pairs = st.get("force_motion_pairs", [])
    if len(pairs) >= 3:
        chain_events = 0
        for i in range(len(pairs) - 1):
            p_cur = pairs[i]
            p_next = pairs[i + 1] if i + 1 < len(pairs) else None
            if p_next is None:
                continue
            cur_vel = p_cur.get("velocity_after", (0, 0))
            cur_obj_id = p_cur.get("object_id", -1)
            next_obj_id = p_next.get("object_id", -1)
            if cur_obj_id != next_obj_id and math.hypot(*cur_vel) > 0.1:
                chain_events += 1
        chain_ratio = chain_events / max(len(pairs) - 1, 1)
        scores.append(chain_ratio * 0.6)

    # Signal 2: Grasp-carry-release (agent uses object as tool to reach goal)
    grasp_events = st.get("grasp_carry_events", [])
    if grasp_events:
        # Each grasp-carry where agent moved object > threshold counts
        carry_count = sum(1 for e in grasp_events if e.get("carry_distance", 0) > 0.3)
        scores.append(min(1.0, carry_count / 3.0))

    # Signal 3: Tool use (agent uses held object to push another)
    tool_events = st.get("tool_use_events", [])
    if tool_events:
        scores.append(min(1.0, len(tool_events) / 2.0))

    # Signal 4: Ordered contact (systematic exploration)
    obj_contact_order = st.get("object_contact_order", [])
    ordered_contacts = _measure_ordered_contact(obj_contact_order)
    if ordered_contacts > 0:
        scores.append(ordered_contacts * 0.4)

    if not scores:
        return 0.0
    # Take the max signal (any one signal passing is enough)
    return min(1.0, max(scores))


def _measure_ordered_contact(order: list) -> float:
    """Measure how ordered the object contact sequence is (0=random, 1=systematic)."""
    if len(order) < 3:
        return 0.0
    # Count transitions between different objects
    transitions = 0
    runs = 1
    for i in range(1, len(order)):
        if order[i] != order[i - 1]:
            transitions += 1
        else:
            runs += 1
    # Higher runs/transitions ratio = more systematic (sticking with objects)
    if transitions == 0:
        return 0.0
    systematic_ratio = min(1.0, runs / max(transitions, 1) / 2.0)
    return systematic_ratio


# --- 里程碑 5: 心智理论 (false-belief) ~ 4 岁 ---
# 评测: agent 能否在物体被遮挡时主动朝其最后已知位置搜索。
# 这是 "理解他者可能存在不同信念" 的前体:
#   → agent 需要在物体不可见时仍保持对其位置的 "信念"
#   → 主动搜索行为 = 信念驱动 (非随机漫游)
#
# 信号: occlusion_events 中 agent 轨迹是否朝 last_known 靠近。
# 仅使用主动搜索行为指标,避免被动物体追踪的自然高准确率。
def _eval_theory_of_mind(st: dict) -> float:
    # Check for explicit score first
    explicit = st.get("tom_score")
    if explicit is not None and explicit > 0:
        return float(explicit)

    occ = st.get("occlusion_events", [])
    if not occ:
        return 0.0

    # 仅用 agent 主动搜索行为: 遮挡期间朝 last_known 靠近的比例
    approach_ratios: list[float] = []
    for ev in occ:
        lk = ev.get("last_known", None)
        traj = ev.get("agent_traj_during_occ", [])
        if lk is None or len(traj) < 2:
            continue
        start_d = _dist(traj[0], lk)
        if start_d < 0.05:
            continue  # already at object, trivial
        end_d = _dist(traj[-1], lk)
        approach = max(0.0, 1.0 - end_d / start_d)
        approach_ratios.append(approach)

    if not approach_ratios:
        return 0.0

    # 至少朝目标靠近 30% 才计入有效搜索
    effective = [a for a in approach_ratios if a > 0.3]
    effective_ratio = len(effective) / len(approach_ratios)

    # 结合平均靠近程度
    mean_approach = float(np.mean(approach_ratios))
    score = effective_ratio * 0.6 + mean_approach * 0.4
    return min(1.0, score)


# --- 里程碑 6: 系统推理 / 守恒 (conservation) ~ 7-11 岁 ---
# 评测: agent 是否表现出系统性的行为模式,而非随机试错。
#
# 三个信号源:
#   1. 动作序列熵 — 低熵 = 有策略,非随机
#   2. 力-动一致性 — 施力与运动方向配对时的一致性
#   3. 规则归纳引擎输出 — rule_engine 发现了多少稳定的环境规则
def _eval_systematic_reasoning(st: dict) -> float:
    # Check for explicit score
    explicit = st.get("systematic_score")
    if explicit is not None and explicit > 0:
        return float(explicit)

    # 1. Action entropy: lower = more systematic
    # Normalize by the ACTUAL action space size (env-provided), not a hardcoded
    # 8 — a 12-action policy is uniformly random at H=ln12 > ln8, which would
    # zero this term even for perfectly systematic behavior (Stage 18 fix).
    actions = st.get("actions", [])
    num_actions = int(st.get("num_actions", 8))
    entropy_score = 0.0
    if len(actions) > 10:
        from collections import Counter
        counts = Counter(actions)
        total = len(actions)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in probs)
        max_entropy = math.log(max(num_actions, 2))  # action space size
        entropy_score = max(0.0, 1.0 - entropy / max_entropy)

    # 2. Force-motion consistency: consistent mapping between force dir and outcome
    pairs = st.get("force_motion_pairs", [])
    fm_consistency = 0.0
    if len(pairs) > 3:
        force_dirs: list[int] = []
        for p in pairs:
            f = p.get("force", (0, 0))
            angle = math.atan2(f[1], f[0])
            # Quantize to 8 direction buckets
            bucket = int((angle + math.pi) / (math.pi / 4)) % 8
            force_dirs.append(bucket)
        from collections import Counter
        dir_counts = Counter(force_dirs)
        if dir_counts:
            most_common_ratio = dir_counts.most_common(1)[0][1] / len(force_dirs)
            fm_consistency = min(1.0, most_common_ratio * 3.0)

    # 3. Rule discovery: number of stable rules from rule_engine
    rules = st.get("rule_count", 0)
    rule_score = min(1.0, rules / 20.0)  # max out at 20 rules

    # Weighted combination — all three must co-signal
    # Only high when ALL dimensions show systematic behavior
    score = entropy_score * 0.35 + fm_consistency * 0.3 + rule_score * 0.35
    # Apply stringent multiplicative gate
    signal_count = sum(1 for x in [entropy_score, fm_consistency, rule_score] if x > 0.1)
    if signal_count < 2:
        score *= 0.3  # heavy penalty if only 1 dimension active
    elif signal_count < 3:
        score *= 0.6  # moderate penalty if 2 dimensions
    return min(1.0, score)


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
        # Estimated age = max age where ALL milestones at or below that age PASS
        for m in sorted(MILESTONES, key=lambda x: x.age_years):
            if report.passed.get(m.key, False):
                max_age = m.age_years
            else:
                break  # younger milestones must all pass
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
        for key in ("occlusion_events", "force_motion_pairs", "count_trials",
                     "actions", "object_contact_order",
                     "grasp_carry_events", "tool_use_events", "release_events"):
            merged = []
            for st in states:
                merged.extend(st.get(key, []))
            agg[key] = merged
        # 透传最新单值分数
        if states:
            last = states[-1]
            for k in ("means_ends_score", "tom_score", "systematic_score",
                       "rule_count", "num_actions"):
                if k in last:
                    agg[k] = last[k]
        return agg


def estimate_cognitive_age(env_states: list[dict]) -> MilestoneReport:
    """便捷函数: 直接给状态序列, 返回报告。"""
    return DevelopmentalEvaluator().evaluate(env_states)
