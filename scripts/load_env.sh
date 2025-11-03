#!/usr/bin/env sh
# Usage:
#   . scripts/load_env.sh             # uses ./.env
#   . scripts/load_env.sh path/to/.env
#
# Exports variables from a .env file into the *current* shell.
# Works in sh/bash/zsh/dash; Linux/mac. No writes to shell rc files.

set -e

ENV_FILE="${1:-.env}"

# Detect whether we're sourced (return works) or executed (return fails)
SOURCED=0
( return 0 2>/dev/null ) && SOURCED=1

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found" >&2
  if [ "$SOURCED" -eq 1 ]; then return 1; else exit 1; fi
fi

exported_keys=""

# Read KEY=VALUE lines; ignore blanks/comments; allow 'export KEY=VALUE'
# Quoted values supported.
while IFS= read -r line || [ -n "$line" ]; do
  # trim
  line=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac

  case "$line" in export\ *) line=${line#export };; esac
  case "$line" in *=*) : ;; *) echo "Skipping (no '='): $line" >&2; continue;; esac

  key=${line%%=*}
  val=${line#*=}
  key=$(printf '%s' "$key" | sed -e 's/[[:space:]]*$//')
  val=$(printf '%s' "$val" | sed -e 's/^[[:space:]]*//')

  case "$val" in
    \"*\") val=${val#\"}; val=${val%\"} ;;
    \'*\') val=${val#\'}; val=${val%\'} ;;
  esac

  case "$key" in ''|*[!A-Za-z0-9_]*|[0-9]*)
    echo "Skipping invalid key: $key" >&2; continue;;
  esac

  # Export exactly; no word-splitting
  # shellcheck disable=SC2163
  export "$key=$val"
  exported_keys="$exported_keys $key"
done < "$ENV_FILE"

# Ensure PYTHONPATH includes '.' even if .env forgot it
case ":$PYTHONPATH:" in
  *:.:*) : ;;  # already present
  *) export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}";;
esac

echo "Loaded from $ENV_FILE:"
for k in $exported_keys PYTHONPATH; do
  eval v=\$$k
  printf " - %s=%s\n" "$k" "${v:-<unset>}"
done

# If executed (not sourced), warn the user
if [ "$SOURCED" -eq 0 ]; then
  echo "NOTE: You executed this script. Run '. scripts/load_env.sh' to affect your current shell." >&2
fi

