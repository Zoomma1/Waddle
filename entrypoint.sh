#!/bin/sh
set -u
mkdir -p logs

pids=""

# docker stop envoie SIGTERM à PID 1 (ce shell). /bin/sh ne le forwarde PAS aux
# enfants background → sans ce trap, les 4 bots meurent en SIGKILL brutal en pleine
# écriture SQLite = corruption (WADDLE-14). Ici on relaie SIGTERM aux bots (qui le
# convertissent en arrêt gracieux) puis on attend leur fermeture propre.
term() {
  echo "[entrypoint] SIGTERM reçu → arrêt gracieux des bots"
  for p in $pids; do kill -TERM "$p" 2>/dev/null || true; done
  wait
  exit 0
}
trap term TERM INT

# Démarrage étalé des 4 bots fleet (évite le rate-limit Bybit au boot)
for i in 1 2 3 4; do
  echo "[entrypoint] starting combo${i}"
  uv run scripts/run_bot.py \
    --config config/fleet/combo${i}_bot.yaml \
    --active-params config/fleet/combo${i}_params.json \
    >> logs/fleet_combo${i}.log 2>&1 &
  pids="$pids $!"
  sleep 15
done

# Bot retuné (config/active_params.json, hot-reload par retune.py) — DB data/waddle_bot.db
echo "[entrypoint] starting retuned bot"
uv run scripts/run_bot.py \
  --config config/bot.yaml \
  --active-params config/active_params.json \
  >> logs/bot_retuned.log 2>&1 &
pids="$pids $!"
sleep 15

# Scheduler telegram_push : push de confirmation après warmup, puis toutes les PUSH_INTERVAL_H
PUSH_INTERVAL_H="${PUSH_INTERVAL_H:-6}"
(
  sleep 180
  uv run scripts/telegram_push.py >> logs/telegram_push.log 2>&1 || true
  while true; do
    sleep "$(( PUSH_INTERVAL_H * 3600 ))"
    uv run scripts/telegram_push.py >> logs/telegram_push.log 2>&1 || true
  done
) &
pids="$pids $!"

# Container vivant tant qu'au moins un job tourne ; SIGTERM géré par le trap ci-dessus
wait
