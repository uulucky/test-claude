#!/usr/bin/env python3
"""Claude Messages API load test. Python 3.10+, pip install httpx."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import getpass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit

import httpx


@dataclass
class Result:
    request_id: int
    latency_ms: float
    status: int | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def endpoint(base_url):
    base_url = base_url.rstrip("/")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("base-url 必须是无账号、密码、查询参数的 HTTP(S) 地址")
    if base_url.endswith("/v1/messages"):
        return base_url
    return base_url + ("/messages" if base_url.endswith("/v1") else "/v1/messages")


async def request_once(client, url, payload, request_id, timeout):
    started = time.perf_counter()
    result = Result(request_id=request_id, latency_ms=0)
    try:
        # wait_for limits the entire request, not just each socket operation.
        response = await asyncio.wait_for(client.post(url, json=payload), timeout)
        result.status = response.status_code
        if not response.is_success:
            result.error = f"HTTP_{response.status_code}"
        else:
            data = response.json()
            if (not isinstance(data, dict) or data.get("type") != "message"
                    or not isinstance(data.get("content"), list)):
                raise ValueError("Unexpected Messages response")
            usage = data.get("usage", {})
            tokens = [usage.get("input_tokens", 0), usage.get("output_tokens", 0)]
            if any(type(n) is not int or n < 0 for n in tokens):
                raise ValueError("Invalid usage")
            result.input_tokens, result.output_tokens = tokens
    except (asyncio.TimeoutError, httpx.TimeoutException):
        result.error = "timeout"
    except httpx.RequestError as exc:
        result.error = type(exc).__name__
    except (ValueError, TypeError, AttributeError):
        result.error = "invalid_response"
    # Never log request headers, response bodies, prompts or exception strings.
    result.latency_ms = (time.perf_counter() - started) * 1000
    return result


def percentile(values, quantile):
    return values[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def summarize(results, elapsed, concurrency):
    successful = [r for r in results if r.error is None]
    latencies = sorted(r.latency_ms for r in successful)
    return {
        "requests": len(results),
        "concurrency": concurrency,
        "success": len(successful),
        "failed": len(results) - len(successful),
        "success_rate_pct": round(100 * len(successful) / len(results), 2),
        "elapsed_seconds": round(elapsed, 3),
        "completed_rps": round(len(results) / elapsed, 3),
        "successful_rps": round(len(successful) / elapsed, 3),
        "success_latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "input_tokens": sum(r.input_tokens for r in successful),
        "output_tokens": sum(r.output_tokens for r in successful),
        "http_status_counts": dict(Counter(str(r.status) for r in results if r.status)),
        "errors": dict(Counter(r.error for r in results if r.error)),
    }


async def run(args, api_key, transport=None, progress=True):
    url = endpoint(args.base_url)
    headers = {"x-api-key": api_key, "anthropic-version": args.api_version}
    if args.workspace_id:
        headers["anthropic-workspace-id"] = args.workspace_id
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
    }
    results = []
    request_ids = iter(range(1, args.requests + 1))
    limits = httpx.Limits(max_connections=args.concurrency,
                          max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=args.timeout,
                                follow_redirects=False, transport=transport) as client:
        async def worker():
            for request_id in request_ids:
                result = await request_once(client, url, payload, request_id, args.timeout)
                results.append(result)
                if progress and (len(results) % 10 == 0 or len(results) == args.requests):
                    print(f"\r已完成 {len(results)}/{args.requests}", end="", flush=True)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(min(args.concurrency, args.requests))))
        elapsed = time.perf_counter() - started
    if progress:
        print()
    return summarize(results, elapsed, args.concurrency), sorted(results, key=lambda r: r.request_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Claude API 压测：默认并发 50，不自动重试")
    parser.add_argument("--base-url", default=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "opus5.5"), help="模型 ID，默认 opus5.5，也可设置 ANTHROPIC_MODEL")
    parser.add_argument("--concurrency", type=int, default=50, help="并发数，默认 50")
    parser.add_argument("--requests", type=int, default=100, help="请求总数，默认 100")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120, help="单次请求总超时（秒）")
    parser.add_argument("--prompt", default="封号不退款，是及其恶劣的行为")
    parser.add_argument("--api-version", default="2023-06-01")
    parser.add_argument("--workspace-id", default=os.getenv("ANTHROPIC_WORKSPACE_ID"))
    parser.add_argument("--key-file", type=Path, help="从文件读取 API Key，例如 /etc/key.txt；优先于环境变量")
    parser.add_argument("--output", type=Path, help="可选：保存 JSON 统计及每个请求的指标")
    args = parser.parse_args()
    if not args.model:
        parser.error("请通过 --model 或 ANTHROPIC_MODEL 指定实际可用的模型 ID")
    if min(args.concurrency, args.requests, args.max_tokens) <= 0:
        parser.error("concurrency、requests、max-tokens 必须大于 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout 必须是大于 0 的有限数值")
    try:
        endpoint(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if args.key_file is not None:
        try:
            api_key = args.key_file.expanduser().read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            raise SystemExit("无法读取 Key 文件，请检查路径、读取权限及 UTF-8 编码") from None
        if not api_key:
            raise SystemExit("Key 文件不能为空")
        if any(character.isspace() for character in api_key):
            raise SystemExit("Key 文件只能包含一个 Key，请勿添加变量名、引号或其他内容")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        if not sys.stdin.isatty():
            raise SystemExit("非交互运行请使用 --key-file 或设置环境变量 ANTHROPIC_API_KEY")
        api_key = getpass.getpass("API Key（粘贴后回车，输入不显示）: ").strip()
    if not api_key:
        raise SystemExit("API Key 不能为空")
    print(f"并发 {args.concurrency}，总请求 {args.requests}，每次最多 {args.max_tokens} 输出 token")
    summary, results = asyncio.run(run(args, api_key))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        report = {"summary": summary, "results": [asdict(r) for r in results]}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已保存：{args.output}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
