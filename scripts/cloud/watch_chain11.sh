#!/usr/bin/env bash
# Chain #11 v2 (2026-10-12): CONTINUATION chain. NEVER uses a fixed first
# ckpt — each segment resumes from the LATEST checkpoint (the developmental
# line is never regressed). v2 bugfix (B9): the segment target is extended
# BEFORE each launch (resume step + 250k) — v1 extended it only AFTER a
# segment, so resuming exactly AT total_steps exited in 0.0s ("no progress")
# and the chain died instantly. Gate: min3>=0.25 deep-collapse floor,
# mean5>=0.55 sustained-health floor (chain10_gate.py). Cap 6 segs.
# Stage 22 M4 (concept bridge) active from the first segment; watch for
# "[concept]" telemetry in train_chain11_s*.log.
set -u
cd /root/karbon
LOG=/root/watch_chain11.log
echo "[chain11] started $(date) pid=$$" >> "$LOG"

while pgrep -f "src[.]train" > /dev/null 2>&1; do sleep 30; done
sleep 5

seg=0
while [ $seg -lt 6 ]; do
    if [ -f /root/STOP_CHAIN ]; then echo "[chain11] STOP_CHAIN $(date)" >> "$LOG"; break; fi
    CKPT=$(ls -t /root/karbon/checkpoints/ckpt_stage20_*.pt 2>/dev/null | head -1)
    [ -f "$CKPT" ] || { echo "[chain11] ckpt missing" >> "$LOG"; break; }
    seg=$((seg+1))
    STEP=$(basename "$CKPT" | grep -oE '[0-9]{6,}' | head -1); STEP=$((10#$STEP))
    BK=/root/autodl-tmp/karbon_ckpts/backup_stage20_$(printf '%07d' $STEP).pt
    [ -f "$BK" ] || cp "$CKPT" "$BK" 2>>"$LOG"
    ls -t /root/karbon/checkpoints/ckpt_stage20_*.pt 2>/dev/null | tail -n +7 | xargs -r rm -f
    # B9 fix: ensure the segment has room BEFORE launching (target = step+250k)
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
    print(f'[chain11] total_steps {cur} -> {tgt}')
else:
    print(f'[chain11] total_steps {cur} already >= {tgt}, keeping')
PYEOF
    echo "[chain11] seg $seg resume $CKPT (step=$STEP) $(date)" >> "$LOG"
    env MUJOCO_GL=egl DEVAGI_NO_COMPILE=1 .venv/bin/python -u -m src.train \
        --config stage20_hypothesis_v2.yaml --preset cloud_24g --resume "$CKPT" \
        >> /root/train_chain11_s${seg}.log 2>&1 < /dev/null
    RC=$?
    echo "[chain11] seg $seg ended rc=$RC $(date)" >> "$LOG"
    ENDCKPT=$(ls -t /root/karbon/checkpoints/ckpt_stage20_*.pt 2>/dev/null | head -1)
    [ -f "$ENDCKPT" ] || ENDCKPT="$CKPT"
    ENDSTEP=$(basename "$ENDCKPT" | grep -oE '[0-9]{6,}' | head -1); ENDSTEP=$((10#$ENDSTEP))
    if [ "$ENDSTEP" -le "$STEP" ]; then echo "[chain11] no progress (end=$ENDSTEP start=$STEP), stop" >> "$LOG"; break; fi
    MUJOCO_GL=egl DEVAGI_NO_COMPILE=1 nice -n 19 .venv/bin/python -u \
        scripts/eval/op_failure_diag.py --ckpt "$ENDCKPT" --episodes 20 --seed 42 \
        --out /root/op_diag_chain11_${ENDSTEP}.json > /root/op_diag_chain11_${ENDSTEP}.log 2>&1
    RATE=$(grep -a OVERALL /root/op_diag_chain11_${ENDSTEP}.log | tail -1 | grep -oE '"pass_rate": [0-9.]+' | grep -oE '[0-9.]+')
    GATE=$(.venv/bin/python /root/chain10_gate.py "$RATE")
    AVG=$(echo "$GATE" | awk '{print $1}')
    MN=$(echo "$GATE" | awk '{print $2}')
    VERD=$(echo "$GATE" | awk '{print $3}')
    echo "[chain11] gate step=$ENDSTEP rate=$RATE mean5=$AVG min3=$MN -> $VERD" >> "$LOG"
    if [ "$VERD" != "PASS" ]; then
        echo "[chain11] OP GATE FAILED at $ENDSTEP (rate=$RATE mean5=$AVG min3=$MN) -- stopping for review" >> "$LOG"
        break
    fi
done
echo "[chain11] finished $(date)" >> "$LOG"
