#!/bin/bash
# run_forever.sh — keep the 3-way batch going across the whole 1000-uid shard.
# Waits for the current uid_0..49 run to finish, extracts the rest of the data,
# then runs batch_3way over the full range with crash-resume (batch_3way skips
# uids whose json already exists without errors).
# NOTE: no `set -u` — cpedit_env.sh expands $PYTHONPATH/$CPLUS_INCLUDE_PATH which
# may be unset, and that would abort the script under `set -u`.
cd /workspace/cpedit
: "${PYTHONPATH:=}"; : "${CPLUS_INCLUDE_PATH:=}"; export PYTHONPATH CPLUS_INCLUDE_PATH
source /workspace/cpedit_env.sh

TAR=data/nano3d_raw/v1_100k/editing_assets/part_000.tar.gz
LOG=outputs/batch3/run_forever.log
MAXU=999
mkdir -p outputs/batch3

echo "[driver] $(date -u +%F_%H:%M) waiting for current run to finish..." | tee -a "$LOG"
while pgrep -f "scripts/batch_3way.py" >/dev/null; do sleep 60; done

echo "[driver] extracting uid_50..$MAXU data..." | tee -a "$LOG"
MEM=$(mktemp)
for i in $(seq 50 $MAXU); do
  for f in source.png edit_512.png src_mesh.glb tar_mesh.glb; do echo "uid_$i/$f"; done
done > "$MEM"
tar xzf "$TAR" -C data/nano3d -T "$MEM" 2>>"$LOG"
rm -f "$MEM"
echo "[driver] data ready: $(ls -d data/nano3d/uid_* | wc -l) uid dirs" | tee -a "$LOG"

UIDS=$(seq 0 $MAXU | sed 's/^/uid_/')
ATTEMPT=0
while [ $ATTEMPT -lt 80 ]; do
  ATTEMPT=$((ATTEMPT+1))
  DONE=$(ls outputs/batch3/uid_*.json 2>/dev/null | wc -l)
  if [ "$DONE" -ge $((MAXU+1)) ]; then
    echo "[driver] all $((MAXU+1)) json present -> done" | tee -a "$LOG"
    break
  fi
  echo "[driver] === pass $ATTEMPT  $(date -u +%H:%M)  json=$DONE ===" | tee -a "$LOG"
  python scripts/batch_3way.py $UIDS >> "$LOG" 2>&1
  RC=$?
  echo "[driver] pass $ATTEMPT exited rc=$RC json=$(ls outputs/batch3/uid_*.json 2>/dev/null | wc -l)" | tee -a "$LOG"
  # if it exited cleanly with the DONE marker, stop; otherwise resume
  if [ $RC -eq 0 ] && tail -40 "$LOG" | grep -q "### BATCH3 DONE"; then
    echo "[driver] clean finish." | tee -a "$LOG"
    break
  fi
  echo "[driver] no clean finish (rc=$RC) -> resume in 20s" | tee -a "$LOG"
  sleep 20
done
echo "### RUN_FOREVER COMPLETE json=$(ls outputs/batch3/uid_*.json 2>/dev/null | wc -l) $(date -u +%F_%H:%M)" | tee -a "$LOG"
