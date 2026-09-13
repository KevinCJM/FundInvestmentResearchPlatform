#!/usr/bin/env python3
"""Prepare a quality run from live GitHub metadata, using protected workflow code."""
import json
import os
from urllib.parse import quote

from check_ai_review import GitHub, REPOSITORY
from submission_policy import SHA, quality_plan


def main():
    number = int(os.environ["PR_NUMBER"])
    head, base = os.environ["PR_HEAD"], os.environ["PR_BASE"]
    if number <= 0 or not SHA.fullmatch(head) or not SHA.fullmatch(base):
        raise ValueError("Invalid PR or commit identity")
    gh = GitHub()
    pr = gh.api(f"repos/{REPOSITORY}/pulls/{number}")
    if (pr["state"] != "open" or pr["base"]["ref"] not in {"main", "Dev"}
            or any((pr[side].get("repo") or {}).get("full_name") != REPOSITORY for side in ("head", "base"))
            or pr["head"]["sha"] != head):
        raise ValueError("Untrusted or stale PR")
    latest = gh.api(f"repos/{REPOSITORY}/git/ref/heads/{quote(pr['base']['ref'], safe='')}")["object"]["sha"]
    if latest != base:
        raise ValueError("Target advanced; start a new quality run")
    comparison = gh.api(f"repos/{REPOSITORY}/compare/{base}...{head}")
    if comparison["merge_base_commit"]["sha"] != base:
        raise ValueError("Source must contain the latest target")
    # Bind selection to immutable commits, never to the mutable PR /files API.
    # GitHub caps compare.files at 300; at that boundary conservatively run all
    # business suites instead of treating a truncated docs sample as docs-only.
    files = comparison["files"]
    paths = [path for item in files for path in [item["filename"], item.get("previous_filename")] if path]
    if len(files) >= 300:
        paths.append('__comparison_file_limit_requires_full_suite__')
    plan = quality_plan(paths, pr["base"]["ref"])
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in plan.items():
            stream.write(f"{key}={str(value).lower()}\n")
        stream.write(f"head={head}\nbase={base}\n")
    print(json.dumps({"pr": number, "head": head, "base": base, "sampled_files": len(files), "plan": plan}))


if __name__ == "__main__":
    main()
