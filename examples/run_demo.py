#!/usr/bin/env python3
"""End-to-end demo of the MemCatalog query pipeline.

Runs the example conversation in examples/demo_conversation.json through the
whole read/write cycle, question by question:

    classify_query      planner: POINT / AGGREGATE / COMPARE
    build_memory_query   the five layer switches, the fact-triple view, the
                        MoM read, prompt assembly
    commit_query        the MoM write

By default the fact-triple extraction step runs without an LLM, so the FACTS
block comes back empty and the demo reports that. Pass --llm-cmd to plug in a
model; the command is called once per session with the extraction prompt on
stdin and must write the completion to stdout.

    python run_demo.py
    python run_demo.py --llm-cmd 'my-llm-cli --model x'
    python run_demo.py --show-prompt 0        # suppress the full prompts

Nothing here reads any gold answer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

# The two on-disk stores (fact cache, workload log) are env-overridable and
# Default to a local ignored directory unless the caller chooses another path.
# This must happen before importing the package because stores load on import.
os.environ.setdefault("MEMCAT_FACTS_CACHE", str(ROOT / "results" / "facts_cache.json"))
os.environ.setdefault("MEMCAT_QUERY_LOG", str(ROOT / "results" / "query_log.json"))
(ROOT / "results").mkdir(exist_ok=True)

from memcatalog import (  # noqa: E402
    build_memory_query,
    classify_query,
    commit_query,
    flush_stores,
    popularity_score,
)


# --- the one pluggable dependency -------------------------------------------

class CmdGenerator:
    """Wraps an external command as a generate_fn(prompt) -> str callback."""

    def __init__(self, cmd: str) -> None:
        self.cmd = shlex.split(cmd)

    def __call__(self, prompt: str) -> str:
        p = subprocess.run(
            self.cmd, input=prompt,
            capture_output=True, text=True, timeout=600,
            check=False,
        )
        return p.stdout


class EmptyGenerator:
    """Stands in for the LLM when none is configured. Emits no triples."""

    def __call__(self, prompt: str) -> str:
        return ""


# --- helpers ----------------------------------------------------------------

def session_text(turns) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def rule_name(pattern: str) -> str:
    return {"POINT": "POINT (取一个值)",
            "AGGREGATE": "AGGREGATE (枚举 + 计数)",
            "COMPARE": "COMPARE (拉出候选逐维比较)"}.get(pattern, pattern)


def main() -> int:
    ap = argparse.ArgumentParser(description="MemCatalog pipeline demo")
    ap.add_argument("--data", default=str(HERE / "demo_conversation.json"))
    ap.add_argument("--llm-cmd", default=None,
                    help="shell command for fact extraction; reads the prompt on stdin")
    ap.add_argument("--show-prompt", type=int, default=1,
                    help="1 = print the assembled prompt for each question (default)")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    scope = data["scope"]
    texts = [session_text(s["turns"]) for s in data["sessions"]]
    labels = [s["label"] for s in data["sessions"]]
    dates = [s["date"] for s in data["sessions"]]

    generate_fn = CmdGenerator(args.llm_cmd) if args.llm_cmd else EmptyGenerator()
    if not args.llm_cmd:
        print("未配置 LLM (--llm-cmd)：FACTS 层本次不产出三元组，其余各层照常运行。\n")

    print("=" * 78)
    print(f"MemCatalog demo   scope = {scope}   sessions = {len(texts)}")
    print("=" * 78)
    print("session 正文长度：" + ", ".join(f"{l}={len(t)}" for l, t in zip(labels, texts)))

    logged = 0
    for i, item in enumerate(data["questions"], 1):
        q = item["q"]
        hint = item.get("q_type_hint")

        pattern = classify_query(q)
        res = build_memory_query(
            q, texts, labels, dates, generate_fn,
            scope=scope, q_type_hint=hint,
        )
        plan = res.plan
        # MoM is only read when the plan opens the mom switch; mirroring that here
        # keeps the printed map identical to the one the prompt actually used.
        pop = (popularity_score(scope, res.q_entities, q_type=plan.q_type, min_overlap=1)
               if plan.mom else {})

        print()
        print("-" * 78)
        print(f"[{i}] {q}")
        print(f"    算子路由               : {rule_name(pattern)}")
        print(f"    q_type                 : {plan.q_type}   (hint = {hint})")
        print(f"    五个层开关             : matches={plan.matches} revision={plan.revision} "
              f"facts={plan.facts} hard_rev={plan.hard_rev} mom={plan.mom}")
        print(f"    实体集合 entities_used : {plan.entities_used}")
        print(f"    n_matches_sessions     : {plan.n_matches_sessions}")
        print(f"    n_facts_rendered       : {plan.n_facts_rendered}"
              + (f"  (superseded {plan.n_facts_superseded})" if plan.n_facts_superseded else ""))
        print(f"    读 MoM 之前 log 深度   : {logged}")
        print(f"    n_popular_sessions     : {plan.n_popular_sessions}")
        if pop:
            shown = ", ".join(f"{k}:{v}" for k, v in sorted(pop.items()))
            print(f"    popularity 映射        : {shown}")
        else:
            print("    popularity 映射        : {}  (log 里没有与本题实体重叠的条目)")

        if args.show_prompt:
            print(f"    prompt 长度            : {len(res.prompt)} 字符")
            print()
            for line in res.prompt.splitlines():
                if line.startswith("=== ") and len(line) > 60:
                    print("   ", line[:77] + "…")
                else:
                    print("   ", line)

        commit_query(res)
        if plan.mom:
            logged += 1

    print()
    print("=" * 78)
    print(f"MoM workload log: 本次运行共写入 {logged} 条 (scope = {scope})")
    print("每条记录的字段: q_type / q_entities / matched_sessions")
    print("答案文本与正确性信号都不进 log。")
    flush_stores()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
