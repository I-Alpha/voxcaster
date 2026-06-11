#!/usr/bin/env bash
# Live volume control from the shell — no daemon restart needed.
# The volume applies from the next spoken line onwards.
#
# Run:  bash examples/quiet_hours.sh
set -euo pipefail

speak --volume 100 "this is full volume"
sleep 4

speak --volume 40 "and this is quiet hours"
sleep 4

speak --volume 100 "back to normal"
