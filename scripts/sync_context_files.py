"""Regenerate CLAUDE.md, AGENTS.md, and GEMINI.md from the canonical SKILL.md.

The skill body lives once, in skills/experiment-as-argument/SKILL.md. Each agent
CLI reads its own context file (Claude Code: CLAUDE.md, Codex: AGENTS.md,
Gemini CLI: GEMINI.md); these are byte-identical mirrors of the SKILL.md body
with the frontmatter stripped. Run this after editing SKILL.md, or with
--check to verify the mirrors have not drifted.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "experiment-as-argument" / "SKILL.md"
MIRRORS = [ROOT / "CLAUDE.md", ROOT / "AGENTS.md", ROOT / "GEMINI.md"]


def body() -> str:
    return re.sub(r"^---\n.*?\n---\n\n?", "", SKILL.read_text(), flags=re.S)


def main() -> int:
    expected = body()
    if "--check" in sys.argv:
        drifted = [m.name for m in MIRRORS
                   if not m.exists() or m.read_text() != expected]
        if drifted:
            print(f"DRIFT: {', '.join(drifted)} — run scripts/sync_context_files.py")
            return 1
        print("mirrors in sync")
        return 0
    for m in MIRRORS:
        m.write_text(expected)
        print(f"wrote {m.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
