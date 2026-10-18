#!/usr/bin/env bash
# relaunch_night.sh (2026-10-13): clean duplicate/stale watchers + launch the
# full night set. NO_AUTO_STOP flag stays in place (user: do not shut down).
cd /root

# 1) kill duplicates from the quoting-broken launches
for pat in "watch_op_di[g].sh" "watch_syspro[b]e" "watch_auto_sto[p].sh" "watch_chain1[1].sh" "watch_nigh[t].sh"; do
    pids=$(pgrep -f "$pat")
    [ -n "$pids" ] && kill $pids 2>/dev/null
done
sleep 1
rm -f /root/STOP_CHAIN

# 2) relaunch one of each
setsid nohup bash /root/watch_op_diag.sh     >/dev/null 2>&1 </dev/null &
sleep 1
setsid nohup bash /root/watch_sysprobe_v2.sh >/dev/null 2>&1 </dev/null &
sleep 1
setsid nohup bash /root/watch_auto_stop.sh   >/dev/null 2>&1 </dev/null &
sleep 1
setsid nohup bash /root/watch_night.sh       >/dev/null 2>&1 </dev/null &
sleep 3

# 3) report
echo "=== NO_AUTO_STOP present? ==="
ls /root/NO_AUTO_STOP 2>/dev/null && echo yes-keep-on || echo MISSING
echo "=== running watch/chain/train processes ==="
ps -ef | grep -E "watch_|src\.train" | grep -v grep
echo "=== DONE ==="
