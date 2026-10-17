"""Command-line interface for persistent memory lifecycle operations."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from .config import MemoryConfig
from .engine import MemoryEngine
from .models import Turn


def command_callback(command: Optional[str]) -> Optional[Callable[[str], str]]:
    if not command:
        return None
    argv = shlex.split(command)
    if not argv:
        raise ValueError("model command must not be empty")

    def invoke(prompt: str) -> str:
        process = subprocess.run(
            argv,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or "model command failed")
        return process.stdout

    return invoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memcatalog")
    parser.add_argument("--store", default="results/memory.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="create an empty memory store")

    ingest = subparsers.add_parser("ingest", help="append a closed conversation session")
    ingest.add_argument("--user", required=True)
    ingest.add_argument("--session", required=True)
    ingest.add_argument("--timestamp", required=True)
    ingest.add_argument("--file", required=True, help="JSON list of speaker/content turns")

    query = subparsers.add_parser("query", help="compile, ground, and execute a memory query")
    query.add_argument("--user", required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--choices", help="JSON answer-choice list")
    query.add_argument("--snapshot")
    query.add_argument("--max-calls", type=int, default=8)
    query.add_argument("--fact-command")
    query.add_argument("--answer-command")

    report = subparsers.add_parser("report-use", help="record validated evidence use")
    report.add_argument("--user", required=True)
    report.add_argument("--query", required=True)
    report.add_argument("--facts", default="", help="comma-separated returned fact IDs")
    report.add_argument("--incomplete", action="store_true")

    stats = subparsers.add_parser("stats", help="show memory counts for one user")
    stats.add_argument("--user", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = MemoryConfig(
        store_path=args.store,
        max_memory_calls=getattr(args, "max_calls", 8),
    )
    engine = MemoryEngine(
        config,
        fact_extractor=command_callback(getattr(args, "fact_command", None)),
        answer_fn=command_callback(getattr(args, "answer_command", None)),
    )

    if args.command == "init":
        engine.flush()
        print(args.store)
    elif args.command == "ingest":
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        turns = [Turn(str(item["speaker"]), str(item["content"])) for item in payload]
        engine.remember(args.user, args.session, args.timestamp, turns)
        print(json.dumps(engine.summary(args.user), ensure_ascii=False))
    elif args.command == "query":
        choices = json.loads(args.choices) if args.choices else None
        response = engine.query(
            args.user,
            args.question,
            choices=choices,
            snapshot_at=args.snapshot,
        )
        print(
            json.dumps(
                {
                    "query_id": response.query_id,
                    "plan_class": response.plan.plan_class.value,
                    "status": response.result.status.value,
                    "value": response.result.value,
                    "fact_ids": response.result.evidence.fact_ids,
                    "memory_calls": response.memory_calls,
                    "answer": response.answer,
                    "prompt": response.prompt if response.answer is None else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "report-use":
        fact_ids = [item for item in args.facts.split(",") if item]
        engine.report_use(
            args.user,
            args.query,
            fact_ids,
            complete=not args.incomplete,
        )
        print("recorded")
    elif args.command == "stats":
        print(json.dumps(engine.summary(args.user), ensure_ascii=False, indent=2))
    return 0
