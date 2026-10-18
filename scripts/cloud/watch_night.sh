#!/usr/bin/env bash
# watch_night.sh (2026-10-13) — NIGHT MODE continuation chain.
# User directive (02:50): keep the machine busy overnight; do NOT shut down.
# Same segment mechanics as chain11 v2 (resume latest ckpt, +250k pre-extend,
# per-segment diag, backup+prune) — but the op gate is ADVISORY: segments
# continue through the known-recorded deep-low phase (G2; soak5/6 precedent —
# capability claims come from good-phase backups, not from this window).
# Hard stop ONLY on catastrophic op (min3 < 0.08) or /root/STOP_CHAIN.
# Stage 22 M4 (concept bridge) is active; watch "[concept]" in train_night_s*.log.
set -u
cd /root/karbon
LOG=/root/watch_night.log
echo "[night] started $(date) pid=$$" >> "$LOG"

while pgrep -f "src[.]train" > /dev/null 2>&1; do sleep 30; done
sleep 5

seg=0
PREV_RATE=1.0
while [ $seg -lt 6 ]; do
    if [ -f /root/STOP_CHAIN ]; then echo "[night] STOP_CHAIN $(date)" >> "$LOG"; break; fi
    CKPT=$(ls -t /root/karbon/checkpoints/ckpt_stage20_*.pt 2>/dev/null | head -1)
    [ -f "$CKPT" ] || { echo "[night] ckpt missing" >> "$LOG"; break; }
    seg=$((seg+1))
    STEP=$(basename "$CKPT" | grep -oE '[0-9]{6,}' | head -1); STEP=$((10#$STEP))
    BK=/root/autodl-tmp/karbon_ckpts/backup_stage20_$(printf '%07d' $STEP).pt
    [ -f "$BK" ] || cp "$CKPT" "$BK" 2>>"$LOG"
    ls -t /root/karbon/checkpoints/ckpt_stage20_*.pt 2>/dev/null | tail -n +7 | xargs -r rm -f
    TARGET=$((STEP + 250000))
    .venv/bin/python - "$TARGET" <<'PYEOF' >> "$LOG" 2>&1
import re
import sys
tgt = int(sys.argv[1])
p = 'configs/stage20_hypothesis_v2.yaml'
s = open(p).read()
cur = int(re.search(r'total_steps:\s*(\d+)', s).group(1))
if cur < tgt:
    open(p, 'w').write(re.sub(r'total_steps:\s*\d+', f'total_steps: {tgt}', s, count=1))
    print(f'[night] total_steps {cur} -> {tgt}')
else:
    print(f'[night] total_steps {cur} already >= {tgt}')
PYEOF
    echo "[night] seg $seg resume $CKPT (step=$STEP) $(date)" >> "$LOG"
    env MUJOCO_GL=egl DEVAGI_NO_COMPILE=1 .venv/bin/python -u -m src.train \
        --config stage20_hypothesis_v2.yaml --preset cloud_24g --resume "$CKPT" \
        >> /root/train_night_s${seg}.log 2>&1 < /dev/null
    RC=$?
    echo "[night] seg $seg ended rc=$RC $(date)" >> "$LOG"
    ENDCKPT=$(ls -t /root/karbon/checkpoints/ckpt_stage20_*.pt 2>/dev/null | head -1)
    [ -f "$ENDCKPT" ] || ENDCKPT="$CKPT"
    ENDSTEP=$(basename "$ENDCKPT" | grep -oE '[0-9]{6,}' | head -1); ENDSTEP=$((10#$ENDSTEP))
    if [ "$ENDSTEP" -le "$STEP" ]; then echo "[night] no progress (end=$ENDSTEP start=$STEP), stop" >> "$LOG"; break; fi
    MUJOCO_GL=egl DEVAGI_NO_COMPILE=1 nice -n 19 .venv/bin/python -u \
        scripts/eval/op_failure_diag.py --ckpt "$ENDCKPT" --episodes 20 --seed 42 \
        --out /root/op_diag_night_${ENDSTEP}.json > /root/op_diag_night_${ENDSTEP}.log 2>&1
    RATE=$(grep -a OVERALL /root/op_diag_night_${ENDSTEP}.log | tail -1 | grep -oE '"pass_rate": [0-9.]+' | grep -oE '[0-9.]+')
    GATE=$(.venv/bin/python /root/chain10_gate.py "$RATE")
    echo "[night] diag step=$ENDSTEP rate=$RATE | gate(advisory): $GATE" >> "$LOG"
    # v2 (2026-10-13): stop only on TWO CONSECUTIVE segment-end rates < 0.08.
    # v1 used the dense watcher history's min3, which fired on a single noisy
    # mid-segment dip (0.072 @15.80M while the segment end was 0.126).
    CUR="$RATE"; PRV="$PREV_RATE"
    PREV_RATE="$RATE"
    CAT=$(awk -v a="$CUR" -v b="$PRV" 'BEGIN{print (a+0 < 0.08 && b+0 < 0.08) ? 1 : 0}')
    if [ "$CAT" = "1" ]; then
        echo "[night] CATASTROPHIC op (two consecutive seg-end rates < 0.08: $PRV -> $CUR) at $ENDSTEP -- stopping for review" >> "$LOG"
        break
    fi
done
echo "[night] finished $(date)" >> "$LOG"
