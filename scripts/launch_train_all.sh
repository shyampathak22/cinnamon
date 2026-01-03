#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/launch_train_rope.sh" "$@"
"$SCRIPT_DIR/launch_train_rope_yarn.sh" "$@"
"$SCRIPT_DIR/launch_train_pope.sh" "$@"
