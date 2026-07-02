# Full-Journey Plan · Stages 0–6 / 全流程规划

Bilingual. What it takes end-to-end to train the whole `devagi` agent from
skeleton to a 30-day perpetually-running developmental agent.

双语。从零骨架跑到"30 天不重启的发育式 AI"全流程规划。

---

## 1. Decision Matrix / 决策矩阵

Three routes to complete Stages 0–6. Recommendation: **Route C (Hybrid)**.

三条完成 Stage 0–6 的路径。**推荐 Route C（混合）**。

| Route | Stage 0–4 | Stage 5–6 | Total time | Total cost | Risk |
|---|---|---|---|---|---|
| A · Full cloud (vGPU-32) | cloud vGPU | **cloud vGPU** | ~85 days | ¥2900–3500 | Preemption + long-tail |
| B · Full cloud (5090) | cloud 5090 | **cloud 5090** | ~60 days | ¥3900–4550 | Same, more expensive |
| **C · Hybrid** ⭐ | cloud vGPU | **home 64G rig** | ~80 days | ¥1200–1600 + hardware | Lowest |

Route C wins on **cost + perpetuity + long-term ownership**.

---

## 2. Route C Timeline / Route C 时间线

Assumes vGPU-32GB at ¥1.58/hour for cloud, home rig purchased before Stage 5.

### Phase 2: Cloud (vGPU-32GB, ~35 days, ~¥1330)

| Stage | Duration | Cost | Cloud command |
|---|---|---|---|
| 0 · Skeleton + 24 h longevity | 1 d | ¥38 | `bash scripts/cloud/run_stage.sh 0 cloud_5090` |
| 1 · RND + Bounded Replay | 3–5 d | ¥115–190 | `bash scripts/cloud/run_stage.sh 1 cloud_5090` |
| 2a · Hybrid PyTorch | 5–7 d | ¥190–265 | `bash scripts/cloud/run_stage.sh 2 cloud_5090` |
| 2b · Triton hot-swap | 1–2 d | ¥38–75 | with `--config configs/stage2b_ttt_triton.yaml` |
| 3 · Dreamer world model | 7–10 d | ¥265–380 | Crafter env starts here |
| 4 · Bounded Skill Library | 7–10 d | ¥265–380 | LoRA + LRU merging |
| **Cloud total** | **~30 d** | **~¥1000** | — |

**Milestone: `v0.4.0-stage4-cloud` tagged.**

### Migration Day: Cloud → Home 64G (~1 day)

```bash
# On cloud, before shutting down:
bash scripts/cloud/sync_to_git.sh 4 v0.4.0-stage4-cloud
python -m scripts.export_hf \
    --ckpt checkpoints/ckpt_stage4_latest.pt \
    --output-dir exports/devagi-stage4 \
    --arch hybrid_backbone --dtype float16
# Upload exports/ to TOS OR rsync back to home:
rsync -avz --partial checkpoints/ user@home-rig:/home/karbon/checkpoints/

# On home rig:
git clone https://github.com/wuhaodesq/karbon.git
cd karbon
bash scripts/home/setup_env.sh --skip-torch
source .venv/bin/activate
rsync -avz cloud:/karbon/checkpoints/ ./checkpoints/
python -m scripts.preflight --preset home_64g
```

Cloud instance can be destroyed after `rsync` completes.

### Phase 3: Home 64G (~55 days, ~¥275 electricity)

| Stage | Duration | Home command |
|---|---|---|
| 5 · Auto Curriculum (LP) | 6–10 d | `bash scripts/home/run_perpetual.sh 5 home_64g --resume checkpoints/ckpt_stage4_latest.pt` |
| 6a · EWC + GR wiring | 5 d | `bash scripts/home/run_perpetual.sh 6 home_64g` (short exploration) |
| 6b · 30-day perpetuity | **30 d** | Same tmux session left running |
| 6c · Debugging + retries | 15 d | Address any drift or slope alarms |
| **Home total** | **~55 d** | Electricity only |

**Milestone: `v1.0.0-stage6` tagged.**

---

## 3. What "Full Journey" produces / 全流程产出

- ✅ **A running developmental agent** whose weights are frozen for 30+ days without collapse.
- ✅ **~10+ tasks learned in sequence** with ≤10% drop on the earliest tasks (EWC + GR working).
- ✅ **VRAM slope ≈ 0** over 7-day rolling windows (Axiom 5 fully validated).
- ✅ **Git tags v0.0.0 → v1.0.0**, each with a bilingual report and figures.
- ✅ **HF-format exports** of the top-K checkpoints on TOS / HuggingFace Hub / ARK.
- ✅ **A code base** that other researchers can clone, `pytest`, and reproduce.

Not produced:
- ❌ A chatbot / dialogue LLM (out of scope).
- ❌ Frontier-scale (>1B params) results (single-node ceiling).

---

## 4. Home Rig Purchase Guide / 家用机采购建议

Buy **before Stage 5 begins**. Options:

| Config | VRAM | Price (approx) | Notes |
|---|---|---|---|
| RTX 5090 32GB × 1 | 32 GB | ¥16k | Start here; add 2nd card later |
| **RTX 5090 32GB × 2** ⭐ | **64 GB** | ¥32k + system | DDP, but no NVLink |
| RTX 6000 Ada 48GB | 48 GB | ¥45k | Single card, stable |
| A100 80GB (used) | 80 GB | ¥50k+ | Ampere, no sm_120 |
| RTX 4090 24GB × 2 | 48 GB | ¥30k | Common, cheapest DDP |
| RTX 4090 48GB (grey-market mod) | 48 GB | ¥25k | Warranty risk |

**Non-GPU minimums for home rig:**
- CPU: 8+ cores (AMD Ryzen 9 / Intel i9)
- RAM: 128 GB (2× VRAM rule)
- SSD: 2 TB NVMe (replay cold tier)
- PSU: 1000 W+ per GPU

**Total system budget: ¥25k–60k depending on GPU choice.**

---

## 5. Risk Register / 风险登记

| Risk | Route C impact | Mitigation |
|---|---|---|
| vGPU preempted mid-stage | Loss of hours, not the run | Frequent ckpts (every 20k steps); `--resume` |
| Home rig delayed beyond Stage 4 | Extra cloud costs | Preorder GPU; cloud runs Stage 5 as fallback (~¥400 extra) |
| Stage 6 30-day fails | Bounded axiom violation somewhere | `check_bounded` + memory-watcher slope; may downgrade to 7-day |
| Skill library grows unbounded | Axiom 1 breach | Unit tests already assert `len ≤ capacity`; health daemon watches at runtime |
| Home rig power outage during Stage 6 | Ckpt loss up to `ckpt_every_steps` | UPS + auto-resume script |
| Home rig thermal throttle | Slower training | Monitor `nvidia-smi -q -d TEMPERATURE`; underclock or add cooling |

---

## 6. Alternative: Cloud-Only Route A / 备选：全云端

If you can't buy home hardware, Route A is possible but expensive:

- **Stage 6 30-day run**: 720 hours × ¥1.58 = **¥1140** just for the perpetual test.
- **Preemption risk**: on-demand instances can be killed on maintenance; use auto-restart wrapper + frequent ckpts.
- **Recommended cloud config**: **NOT vGPU** for Stage 6 (unstable); switch to dedicated 5090 or A100 for the perpetual run.

Total cloud-only budget: **¥3000–5000** for full journey.

---

## 7. Fastest-to-Stage-4 Sprint / 冲刺 Stage 4 最快路径

If the goal is "how fast can I see Stage 4 results" (not full journey):

- **5090 (¥2.78/hour) × 24/7**: 15–20 days, **¥1000–1330**
- **vGPU (¥1.58/hour) × 24/7**: 25–35 days, **¥950–1330**

Roughly the same cost, half the wall-clock time on 5090. **Pick 5090 if time-sensitive; vGPU if you can wait.**

---

## 8. Decision Checkpoints / 决策节点

Fork points during the journey:

1. **After Stage 2**: is Hybrid outperforming GRU baseline on long-context? If NO, don't proceed to Stage 3 — debug first.
2. **After Stage 3**: does Dreamer world model give ≥3× sample efficiency vs Stage 1? If NO, roll back to Route A of cloud on smaller env.
3. **After Stage 4**: does skill library show reuse across tasks? If NO, Stage 5 will thrash — pause and improve merging.
4. **Migration Day**: home rig must pass `preflight --preset home_64g` before Stage 5 starts. Otherwise, extend cloud rental.
5. **Stage 6 Day 7**: memory-watcher slope should already be flat. If drifting, halt and diagnose.

---

## 9. Recommended Kickoff Sequence / 建议启动顺序

If you decide to go, here's the exact sequence for **the next 8 weeks**:

**Week 1**: Cloud (vGPU-32GB) — Stage 0 skeleton + 24h longevity.
**Week 2**: Stage 1 (RND + replay).
**Week 3–4**: Stage 2a + Stage 2b.
**Week 5**: **Order home rig** (delivery ~1 week).
**Week 5–6**: Stage 3 (world model).
**Week 6–7**: Stage 4 (skill library).
**Week 8**: Migration Day → Stage 5 begins on home rig.

Stage 6 lives on the home rig indefinitely.

---

## 10. Cost / Time Summary / 成本时间总结

| Metric | Route C (Hybrid) |
|---|---|
| Cloud wall-time | ~35 days |
| Home wall-time | ~55 days (30 of which is Stage 6 perpetual) |
| **Total wall-time** | **~90 days (~3 months)** |
| Cloud spend | **¥1200–1600** |
| Home electricity | ~¥275 |
| Home hardware (one-time) | **¥16k–60k** (starts with 1× 5090) |
| Result | A running Stage 6 agent on your desk, continuing to learn forever |

---

## 11. Ready-to-Run Commands (Route C) / 立即可运行

### Cloud kickoff (vGPU-32GB or 5090)

```bash
git clone https://github.com/wuhaodesq/karbon.git
cd karbon
bash scripts/cloud/setup_env.sh --skip-torch
source .venv/bin/activate
python -m scripts.preflight --preset cloud_5090
bash scripts/cloud/run_stage.sh 0 cloud_5090 --smoke-only
bash scripts/cloud/run_stage.sh 0 cloud_5090
bash scripts/cloud/longevity_24h.sh 0 cloud_5090
bash scripts/cloud/sync_to_git.sh 0 v0.0.0-stage0-cloud
```

### After each stage

```bash
bash scripts/cloud/sync_to_git.sh <N> v0.<N>.0-stage<N>-cloud
python -m scripts.export_hf --ckpt checkpoints/ckpt_stage<N>_latest.pt \
    --output-dir exports/devagi-stage<N> \
    --arch hybrid_backbone --dtype float16
```

### Home migration (after Stage 4)

```bash
# On home rig:
git clone https://github.com/wuhaodesq/karbon.git
cd karbon
bash scripts/home/setup_env.sh --skip-torch
source .venv/bin/activate
rsync -avz cloud:/karbon/checkpoints/ ./checkpoints/
python -m scripts.preflight --preset home_64g
bash scripts/home/run_perpetual.sh 5 home_64g \
    --resume checkpoints/ckpt_stage4_latest.pt
```

### Stage 6 perpetual

```bash
# In one tmux session on home rig:
bash scripts/home/run_perpetual.sh 6 home_64g \
    --resume checkpoints/ckpt_stage5_latest.pt

# In another tmux session on home rig:
bash scripts/home/health_daemon.sh logs/health/stage6/
```

Leave both running. Check on it once a day. Sync to Git weekly.
