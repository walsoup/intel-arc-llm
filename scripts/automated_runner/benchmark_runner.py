#!/usr/bin/env python3
"""
Local LLM Tri-Domain Benchmark Runner (Coding, Tool Calling, Creative Writing)
Zero-dependency, portable benchmark execution tool for llama-server router setups.

Usage:
    python scripts/benchmark_runner.py --models Qwable-9B-Claude-Fable-5-OBLITERATED.Q4_K_M
    python scripts/benchmark_runner.py --models model1 model2 --categories coding tools
    python scripts/benchmark_runner.py --all-models
    python scripts/benchmark_runner.py --list-models
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_DATASET = Path(__file__).parent / "benchmark_dataset_30.json"
DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "benchmark_results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run standardized Tri-Domain Benchmark (Coding, Tools, Creative) on local LLM models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="One or more model IDs to benchmark (must match router model names or GGUF names).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark all .gguf models discovered in ./Models directory sequentially.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Query server or Models/ directory and list all available model IDs, then exit.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER,
        help=f"Base URL of llama-server router (default: {DEFAULT_SERVER}).",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Path to benchmark dataset JSON file (default: {DEFAULT_DATASET.name}).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["coding", "tools", "creative", "all"],
        default=["all"],
        help="Filter tasks by domain: coding, tools, creative, or all (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save raw results and benchmark reports (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds per inference request (default: 120s).",
    )
    parser.add_argument(
        "--delay-between-models",
        type=int,
        default=5,
        help="Seconds to pause between swapping models to allow clean router deallocation (default: 5s).",
    )
    return parser.parse_args()


def get_available_models(server_url):
    """Attempt to query router /v1/models endpoint, fallback to scanning Models/ dir."""
    models = []
    try:
        req = urllib.request.Request(f"{server_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", [])]
    except Exception:
        pass

    if not models:
        models_dir = Path(__file__).parent.parent / "Models"
        if models_dir.exists():
            for f in sorted(models_dir.glob("*.gguf")):
                # strip .gguf for router alias
                models.append(f.stem)
    return models


def load_dataset(dataset_path, categories_filter):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    if "all" not in categories_filter:
        cat_map = {"coding": "Coding", "tools": "Tool Calling", "creative": "Creative Writing"}
        allowed = {cat_map[c.lower()] for c in categories_filter if c.lower() in cat_map}
        tasks = [t for t in tasks if t.get("category") in allowed]

    return data, tasks


def send_chat_completion(server_url, model_name, task, timeout):
    endpoint = f"{server_url}/v1/chat/completions"
    messages = [{"role": "user", "content": task["prompt"]}]

    payload = {
        "model": model_name,
        "messages": messages,
        **task.get("params", {}),
    }

    if "tools" in task:
        payload["tools"] = task["tools"]

    req_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_bytes,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            t1 = time.time()
            body = json.loads(res.read().decode("utf-8"))
            choice = body.get("choices", [{}])[0]
            message = choice.get("message", {})
            timings = body.get("timings", {})
            usage = body.get("usage", {})

            return {
                "success": True,
                "duration_wall_sec": round(t1 - t0, 3),
                "finish_reason": choice.get("finish_reason"),
                "content": message.get("content", ""),
                "reasoning_content": message.get("reasoning_content", ""),
                "tool_calls": message.get("tool_calls", []),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "prompt_ms": timings.get("prompt_ms", 0),
                "predicted_ms": timings.get("predicted_ms", 0),
                "tokens_per_second": round(timings.get("predicted_per_second", 0), 2),
                "error": None,
            }
    except Exception as e:
        t1 = time.time()
        return {
            "success": False,
            "duration_wall_sec": round(t1 - t0, 3),
            "finish_reason": "error",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [],
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "prompt_ms": 0,
            "predicted_ms": 0,
            "tokens_per_second": 0,
            "error": str(e),
        }


def heuristic_evaluate(task, result):
    """
    Compute objective heuristic sub-scores and checks.
    For tool calls: verify tool name, args match, and negative tests.
    For coding/creative: verify non-empty, length, and lack of repetition loops.
    """
    category = task["category"]
    c_text = result.get("content", "").strip()
    r_text = result.get("reasoning_content", "").strip()
    t_calls = result.get("tool_calls", [])
    tokens = result.get("completion_tokens", 0)
    finish = result.get("finish_reason")

    score = 0.0
    notes = []

    if not result.get("success"):
        return 0.0, f"Request error: {result.get('error')}"

    # Check repetition loops
    full_text = c_text + " " + r_text
    is_loop = False
    for loop_token in ["</function>", "-1\n-1", "</>", "an an an", "la de la de"]:
        if full_text.count(loop_token) > 5:
            is_loop = True
            notes.append(f"Severe repetition loop detected ('{loop_token}')")
            break

    if is_loop:
        return 0.10, "; ".join(notes)

    # Check premature EOS
    if tokens <= 2 and finish == "stop" and not c_text and not r_text:
        return 0.00, "Immediate EOS termination (empty output)"

    if category == "Tool Calling":
        expected_tool = task.get("expected_tool")
        expected_args = task.get("expected_args", {})

        if expected_tool is None:
            # Negative test: should NOT call tools
            if not t_calls:
                score = 10.0
                notes.append("Correct negative test handling (no tools called)")
            else:
                score = 3.0
                notes.append(f"Failed negative test: called {t_calls[0].get('function', {}).get('name')}")
        else:
            # Positive tool call expected
            if t_calls:
                fn = t_calls[0].get("function", {})
                call_name = fn.get("name")
                args_str = fn.get("arguments", "{}")

                if call_name == expected_tool:
                    score += 5.0
                    notes.append(f"Correct tool '{call_name}'")
                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                        matches = 0
                        for k, v in expected_args.items():
                            if k in args_obj:
                                if str(args_obj[k]).lower() == str(v).lower():
                                    matches += 1
                        if expected_args:
                            arg_ratio = matches / len(expected_args)
                            score += 4.5 * arg_ratio
                            notes.append(f"Parameter accuracy: {matches}/{len(expected_args)}")
                        else:
                            score += 4.5
                        score += 0.5  # JSON validity
                    except json.JSONDecodeError:
                        score += 1.0
                        notes.append("Malformed arguments JSON")
                else:
                    score = 2.0
                    notes.append(f"Wrong tool selected: '{call_name}' expected '{expected_tool}'")
            elif finish == "length":
                score = 7.5
                notes.append("Formulated tool call reasoning but hit max_tokens limit")
            else:
                score = 0.5
                notes.append("Did not emit tool call")

    elif category == "Coding":
        # Baseline check for reasoning + code logic
        if r_text:
            score += 4.0
            notes.append("Engaged reasoning trace")
        if c_text:
            score += 3.5
            notes.append("Generated code body")
            if "def " in c_text or "class " in c_text or "SELECT " in c_text.upper():
                score += 1.5
                notes.append("Syntactically structured code")
        elif finish == "length" and len(r_text) > 300:
            score += 3.5
            notes.append("Detailed algorithmic thought trace (hit token limit)")

        if tokens > 50:
            score += 0.8
        score = min(10.0, max(0.0, score))

    elif category == "Creative Writing":
        if c_text:
            score += 5.0
            notes.append("Direct prose generation")
            if len(c_text.split()) > 40:
                score += 2.5
        elif r_text and finish == "length":
            score += 5.5
            notes.append("Elaborate narrative design in thought trace")

        if tokens > 50:
            score += 2.0

        # Special constraint check: W08 (No 'E')
        if task["id"] == "W08":
            target = c_text or r_text
            if "e" in target.lower():
                score = max(3.0, score - 3.0)
                notes.append("Violated negative constraint (contained letter 'e')")
            else:
                score += 0.5
                notes.append("Respected lipogram constraint")

        score = min(10.0, max(0.0, score))

    return round(score, 2), "; ".join(notes) if notes else "Executed normally"


def run_benchmark():
    args = parse_args()

    available = get_available_models(args.server_url)

    if args.list_models:
        print("\n=== Available Models in System ===")
        if available:
            for idx, m in enumerate(available, 1):
                print(f"  [{idx}] {m}")
        else:
            print("  No models currently detected in router or Models/ directory.")
        print()
        sys.exit(0)

    target_models = []
    if args.all_models:
        target_models = available
    elif args.models:
        target_models = args.models
    else:
        print("Error: Specify models via `--models <name...>` or use `--all-models`.")
        print(f"Discovered models: {', '.join(available) if available else 'None'}")
        sys.exit(1)

    dataset_meta, tasks = load_dataset(args.dataset, args.categories)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"benchmark_run_{timestamp}.json"
    report_file = output_dir / f"benchmark_report_{timestamp}.md"

    print("================================================================")
    print(" Local LLM Tri-Domain Benchmark Runner ")
    print(f" Server:    {args.server_url}")
    print(f" Models:    {len(target_models)} ({', '.join(target_models)})")
    print(f" Tasks:     {len(tasks)} tasks")
    print(f" Results:   {out_file.name}")
    print("================================================================\n")

    benchmark_db = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server_url": args.server_url,
        "dataset_name": dataset_meta.get("name"),
        "models": target_models,
        "results": {},
        "scores": {},
    }

    for m_idx, model in enumerate(target_models, 1):
        print(f"\n[{m_idx}/{len(target_models)}] Benchmarking Model: {model}")
        benchmark_db["results"][model] = {}
        benchmark_db["scores"][model] = {
            "tasks": {},
            "category_scores": {"Coding": [], "Tool Calling": [], "Creative Writing": []},
        }

        for t_idx, task in enumerate(tasks, 1):
            tid = task["id"]
            cat = task["category"]
            title = task["title"]
            print(f"  [{t_idx:02d}/{len(tasks):02d}] {tid} ({cat[:5]}) {title[:28]:<28} ... ", end="", flush=True)

            res = send_chat_completion(args.server_url, model, task, args.timeout)
            score, notes = heuristic_evaluate(task, res)

            benchmark_db["results"][model][tid] = {**task, **res}
            benchmark_db["scores"][model]["tasks"][tid] = {
                "score": score,
                "notes": notes,
                "category": cat,
                "tokens": res["completion_tokens"],
                "speed": res["tokens_per_second"],
                "latency": res["duration_wall_sec"],
            }
            benchmark_db["scores"][model]["category_scores"][cat].append(score)

            # Live status line
            status = f"Score: {score:5.2f} | {res['completion_tokens']:3d} tok | {res['tokens_per_second']:4.1f} t/s | {res['duration_wall_sec']:5.1f}s"
            if not res["success"]:
                status = f"FAILED ({res['error']})"
            print(status)

            # Auto-save incrementally
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_db, f, indent=2)

        if m_idx < len(target_models):
            print(f"  Pausing {args.delay_between_models}s before swapping to next model...")
            time.sleep(args.delay_between_models)

    # Compile Markdown Summary Report
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write(f"# Local LLM Benchmark Report ({timestamp})\n\n")
        rf.write(f"- **Server**: `{args.server_url}`\n")
        rf.write(f"- **Tasks Executed**: {len(tasks)}\n\n")
        rf.write("## Grand Leaderboard\n\n")
        rf.write("| Model | Coding (/10) | Tools (/10) | Creative (/10) | Overall Score | Avg Speed |\n")
        rf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

        leaderboard = []
        for model in target_models:
            cat_s = benchmark_db["scores"][model]["category_scores"]
            c_avg = sum(cat_s["Coding"]) / max(1, len(cat_s["Coding"]))
            t_avg = sum(cat_s["Tool Calling"]) / max(1, len(cat_s["Tool Calling"]))
            w_avg = sum(cat_s["Creative Writing"]) / max(1, len(cat_s["Creative Writing"]))
            all_vals = [s["score"] for s in benchmark_db["scores"][model]["tasks"].values()]
            tot = sum(all_vals) / max(1, len(all_vals))
            speeds = [s["speed"] for s in benchmark_db["scores"][model]["tasks"].values() if s["speed"] > 0]
            avg_speed = sum(speeds) / max(1, len(speeds))
            leaderboard.append((tot, model, c_avg, t_avg, w_avg, avg_speed))

        leaderboard.sort(reverse=True)
        for tot, model, c_avg, t_avg, w_avg, avg_speed in leaderboard:
            rf.write(f"| **{model}** | `{c_avg:.2f}` | `{t_avg:.2f}` | `{w_avg:.2f}` | **`{tot:.2f}`** | {avg_speed:.2f} t/s |\n")

        rf.write("\n\n---\n*Raw evaluation data saved to: `" + out_file.name + "`*\n")

    print("\n================================================================")
    print(" Benchmark Completed!")
    print(f" JSON Dataset: {out_file}")
    print(f" Report:       {report_file}")
    print("================================================================\n")


if __name__ == "__main__":
    run_benchmark()
