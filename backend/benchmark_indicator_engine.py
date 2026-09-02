"""Run one reproducible local benchmark against a persisted evaluation plan."""

from __future__ import annotations

import argparse
import json
import time

from custom_indicators.service import CustomIndicatorService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_id")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--repeat", type=int, default=2)
    arguments = parser.parse_args()

    service = CustomIndicatorService()
    service.start_compute_engine()
    try:
        runs = []
        for index in range(max(1, arguments.repeat)):
            if index == 0:
                service.plan_cache.clear()
            started = time.perf_counter()
            result = service.run_plan(arguments.plan_id, arguments.as_of)
            runs.append(
                {
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "ranked_count": result.get("ranked_count"),
                    "excluded_count": result.get("excluded_count"),
                    "execution": result.get("execution"),
                }
            )
        print(json.dumps({"runs": runs}, ensure_ascii=False, indent=2))
    finally:
        service.close_compute_engine()


if __name__ == "__main__":
    main()
