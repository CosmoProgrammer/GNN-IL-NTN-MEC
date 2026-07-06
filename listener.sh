#!/bin/bash
# listener.sh

BRANCH="main"
INTERVAL=300
TRIGGER_FILE="run_experiments.sh"

echo "Started listener. Polling origin/$BRANCH every $INTERVAL seconds..."

while true; do
    git fetch origin $BRANCH -q

    DIFF=$(git diff --name-only HEAD origin/$BRANCH -- $TRIGGER_FILE)

    if [ "$DIFF" == "$TRIGGER_FILE" ]; then
        echo "[$(date)] Trigger file updated. Pulling changes..."
        git pull origin $BRANCH -q

        echo "[$(date)] Executing $TRIGGER_FILE..."
        chmod +x $TRIGGER_FILE
        nohup ./$TRIGGER_FILE > launch.log 2>&1 &
    fi

    sleep $INTERVAL
done