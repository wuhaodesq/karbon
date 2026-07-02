# Design Principles / 设计公理

Six axioms. Non-negotiable. Every PR must be reviewable against these.
六条公理。不可协商。任何 PR 都要过这六条审查。

---

## Axiom 1 · Zero Unbounded Structures on GPU / GPU 零无界结构

**EN.** No `list.append(...)`-style growth is allowed on GPU-resident state.
Every device-side buffer declares a *capacity* at construction time.
Growth of state is realized only through **eviction of older entries**.

**中.** GPU 上不允许任何 append-only 数据结构。所有设备驻留 buffer 必须在构造时**显式声明容量**。想"记住更多"只能通过**淘汰旧的**实现。

*Checks*
- Static: `scripts/ci/check_bounded.py` scans for suspicious patterns.
- Runtime: every bounded module asserts `len(self) <= self.capacity`.

---

## Axiom 2 · Eviction Before Learning / 淘汰先于学习

**EN.** Before ingesting a new item into any bounded store, the store must decide *what to drop*. Growth is never grow-only.

**中.** 学新东西前先决定"忘什么"。绝不允许只增不减。

*Applies to*: replay buffer, skill library, Fisher matrices, curriculum task pool, world-model imagination cache.

---

## Axiom 3 · Enforced Hierarchical Storage / 强制分层存储

**EN.** Four tiers with explicit demotion policies:
- **Hot**: on-GPU, latency-critical, tiny fixed capacity.
- **Warm**: on-CPU RAM, larger cap, used for near-term rehearsal.
- **Cold**: on SSD/disk, mostly-archival, occasionally sampled.
- **Archive**: off-machine (object storage / cold rsync target), retrieved by hand.

**中.** 四级分层：GPU（热）/ CPU（温）/ SSD（冷）/ 离机归档。每一级都有明确降级策略。

*Anti-pattern*: keeping all history on GPU "because it's fast".

---

## Axiom 4 · Periodic Consolidation (Sleep) / 周期性固化（睡眠）

**EN.** Transient state (TTT inner weights `W`, fresh replay entries, skill candidates, Fisher deltas) must be periodically **distilled into fixed slow-changing parameters**, then cleared.

This mirrors biological sleep — during "night", the agent stops interacting and reorganizes memory.

**中.** 流动状态必须定期蒸馏进"慢权重"，然后清空——类似人做梦时整理记忆。

*Trigger*: every `N` steps (configurable), or when transient state approaches its budget.

---

## Axiom 5 · Fragmentation Governance / 碎片治理

**EN.** Long-running training on modern PyTorch inevitably fragments allocator arenas even when logical usage is flat. This project mitigates via:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set globally.
2. Active `torch.cuda.empty_cache()` every `K` steps.
3. `src/monitoring/memory_watcher.py` monitors *slope of VRAM usage over a rolling 1h window* and alerts if the slope exceeds threshold.

**中.** 长期训练必然产生显存碎片。必须内嵌监控 + 主动 empty_cache + 允许碎片治理策略。**不允许通过重启进程当兜底**（那违反 Axiom 6 的精神）。

---

## Axiom 6 · State is Serializable (for Upgrade, not Restart) / 状态可序列化（为升级，不为重启）

**EN.** Every component (model, optimizer, replay, skill library, curriculum, EWC state, RNG) must be checkpointable and restorable.

The purpose is **upgrade & migration** (e.g., cloud → home-64G), **not** "restart to clear a leak". Using restart as a workaround is a violation of the perpetual-runtime target.

**中.** 每个组件（模型/优化器/回放/技能库/课程/EWC/RNG）都必须可 checkpoint、可恢复。目的是**升级和迁移**，不是"跑挂了重启一下就好"。用重启当兜底 = 违反永续运行目标。

---

## Enforcement Layers / 强制层

| Layer | What it does |
|---|---|
| **Static** | `scripts/ci/check_bounded.py` grep-based lint for suspicious append patterns |
| **Runtime** | every bounded class asserts `len(self) <= self.capacity` in `__len__` and after each mutation |
| **Monitoring** | `src/monitoring/memory_watcher.py` fails a run if VRAM/RAM slope exceeds threshold over 1h |
| **Review** | any deviation must be justified in the PR body, quoting the specific axiom |

---

## Escape Hatches / 例外机制

An axiom violation is permitted **only** when:
1. It is temporary (marked with a `# BOUNDS-EXCEPTION` comment tagged with an issue #).
2. It is explicitly reviewed and time-boxed.
3. It has an owner and a removal deadline.

Otherwise: no exceptions.
