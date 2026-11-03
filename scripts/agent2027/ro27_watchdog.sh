#!/usr/bin/env bash
# AGENT2027 watchdog: read-only checks every 15 min, Google Chat ping on trouble.
# Webhook URL is read from ~/.ro27_webhook (chmod 600). Stop: touch ~/ro27_watchdog_STOP.
# Alerts only when Henrik is needed: GPU 1-7 empty for 30 min; queue runners/launcher gone while
# sessions are still planned; disk > 90 %. No routine status messages.
set -u
ST=~/ro27_watchdog; mkdir -p "$ST"
URL=$(cat ~/.ro27_webhook)
PLANNED="S1027a S1027b S1027c S1027d S1027g S1027h S1027i S1027j S1027k S1027l S1027m S1027n S1028a S1028b S1029a S1029b S1029c"

send() {  # one-line-per-\n text, no double quotes
  local t; t=$(printf '%s' "$1" | sed 's/"/'"'"'/g' | awk 'BEGIN{ORS="\\n"}{print}')
  curl -s -m 20 -X POST "$URL" -H 'Content-Type: application/json; charset=utf-8' \
       -d "{\"text\": \"$t\"}" >/dev/null || true
  echo "[$(date -u +%FT%TZ)] SENT: $1" >> "$ST/watchdog.log"
}

counts() {  # SID -> "ok/failed/cap"
  local f=~/ro27_$1.job.log
  [ -f "$f" ] || { echo "not started"; return; }
  local ok fa ca
  ok=$(grep -cE '\] ROW [^ ]+ attempt=[0-9]+ status=OK' "$f")
  fa=$(grep -cE '\] ROW [^ ]+ attempt=[0-9]+ status=FAILED' "$f")
  ca=$(grep -cE '\] ROW [^ ]+ attempt=[0-9]+ status=CAP_HIT' "$f")
  grep -q "RO27_SESSION_END $1 " "$f" && echo "ended: OK $ok, failed $fa, cap $ca" \
    || echo "running: OK $ok, failed $fa, cap $ca"
}

summary() {
  local s out=""
  for s in $PLANNED; do out="$out
$s $(counts $s)"; done
  echo "$out"
}

while [ ! -e ~/ro27_watchdog_STOP ]; do
  now=$(date -u +%FT%H:%MZ)

  # 1. GPUs 1-7 empty (<1 GiB) for two consecutive checks; one alert per idle episode
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
  while IFS=', ' read -r g mem; do
    [ "$g" = 0 ] && continue
    if [ "${mem:-0}" -lt 1000 ]; then
      n=$(( $(cat "$ST/idle$g" 2>/dev/null || echo 0) + 1 )); echo $n > "$ST/idle$g"
      [ $n -eq 2 ] && send "⚠️ RO27: GPU $g has been empty for 30 min ($now). Queue may be stuck or work is done."
    else
      echo 0 > "$ST/idle$g"
    fi
  done

  # 2. runners gone while planned sessions have not ended
  open=""
  for s in $PLANNED; do grep -qsE "RO27_SESSION_END $s |RO27_CLIENTS_DONE" ~/ro27_$s.job.log ~/ro27_$s.out || open="$open $s"; done
  procs=$(pgrep -fc 'ro27_queue.sh|ro27_launch_A19r.sh|ro27_launch_amend19c.sh|ro27_clients_manifest.sh' || true)
  if [ -n "$open" ] && [ "${procs:-0}" -eq 0 ]; then
    [ -e "$ST/norunner" ] || { touch "$ST/norunner"; send "🚨 RO27: no queue runner or launcher is alive, but these sessions have not ended:$open ($now). Needs a restart."; }
  else
    rm -f "$ST/norunner"
  fi

  # 3. disk
  use=$(df --output=pcent ~ | tail -1 | tr -dc 0-9)
  if [ "${use:-0}" -gt 90 ]; then
    [ -e "$ST/disk" ] || { touch "$ST/disk"; send "⚠️ RO27: /raid is ${use}% full ($now)."; }
  fi

  sleep 900
done
