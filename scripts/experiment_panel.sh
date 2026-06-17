#!/usr/bin/env bash
# Independent multi-model review of an ML experiment plan or result.
#
# Sends the same experiment to several models (claude/codex/gemini), each
# reviewing independently, then synthesizes consensus vs disagreement.
# Disagreement is the signal — the most uncertain claims, worth looking at
# first (see SKILL.md "Independent Review").
#
# Usage:
#   PANELISTS="claude:opus codex:gpt-5 gemini:gemini-2.5-pro" \
#     ./scripts/experiment_panel.sh <plan-or-result-file>
#
# Each PANELISTS entry is "<cli>:<model>". The synthesis step uses claude.
set -uo pipefail

PANELISTS="${PANELISTS:-claude:opus codex:gpt-5 gemini:gemini-2.5-pro}"
SYNTH="${SYNTH:-claude:opus}"
FILE="${1:?usage: experiment_panel.sh <plan-or-result-file>}"
[ -f "$FILE" ] || { echo "no such file: $FILE" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/../.panel/$(date +%Y%m%d_%H%M%S)_$$"  # gitignored work dir
mkdir -p "$OUT"

REVIEW_PROMPT='You are independently reviewing the ML experiment below (a plan or a result) for rigor. In at most 8 bullets, flag: data leakage risks, unsupported or overstated claims, missing baselines/controls, seed/variance gaps, and anything that would stop the result from reproducing. Be specific and concise. Do not rewrite the experiment.'

SYNTH_PROMPT='Below are independent reviews of the SAME ML experiment from different models. Produce three short sections: (1) CONSENSUS — points most reviewers agree on; (2) DISAGREEMENT — points where they diverge; these are the most uncertain and should be investigated first; (3) TOP 3 ACTIONS. Be concise and do not invent points no reviewer raised.'

# Drive one panelist CLI non-interactively; prompt+experiment arrive on stdin.
ask_panelist() {
  local cli="$1" model="$2"
  case "$cli" in
    claude) claude --print --model "$model" \
              --setting-sources local --dangerously-skip-permissions ;;
    codex)  codex exec --model "$model" --sandbox read-only --skip-git-repo-check - ;;
    gemini) gemini --model "$model" --approval-mode plan -p "$(cat)" ;;
    *)      echo "unknown panelist cli: $cli" >&2; return 99 ;;
  esac
}

got=""
for entry in $PANELISTS; do
  cli="${entry%%:*}"; model="${entry#*:}"
  echo ">> panelist $cli:$model"
  if { printf '%s\n\n--- EXPERIMENT ---\n' "$REVIEW_PROMPT"; cat "$FILE"; } \
       | ask_panelist "$cli" "$model" > "$OUT/$cli.txt" 2> "$OUT/$cli.stderr" \
       && [ -s "$OUT/$cli.txt" ]; then
    got="$got $cli"
  else
    echo "!! panelist $cli failed or returned nothing (see $OUT/$cli.stderr)" >&2
  fi
done

[ -n "$got" ] || { echo "no panelist returned a review" >&2; exit 1; }

synth_cli="${SYNTH%%:*}"; synth_model="${SYNTH#*:}"
{
  printf '%s\n' "$SYNTH_PROMPT"
  for cli in $got; do printf '\n=== reviewer: %s ===\n' "$cli"; cat "$OUT/$cli.txt"; done
} | ask_panelist "$synth_cli" "$synth_model" | tee "$OUT/synthesis.md"

echo "panel complete (reviewers:$got): $OUT/synthesis.md"
