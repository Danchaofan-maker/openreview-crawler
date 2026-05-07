#!/usr/bin/env python3
"""DeepSeek V4 打分测试 — 用 prompt.md 给 sample_20 中前 N 篇论文打分

用法: uv run scripts/score_paper_test.py [N] [model]
默认 N=3, model=deepseek-v4-pro (旗舰 hybrid 模型,thinking 默认开)
环境变量: DEEPSEEK_API_KEY 必填

请求中显式传 thinking + reasoning_effort,确保 reasoning 通道开启;
响应里 reasoning_content 与 content 同级,前者保存为日志,后者必须是纯 JSON。
"""
import os
import sys
import json
import time
from pathlib import Path
import requests

PROMPT_PATH = "prompt.md"
SAMPLE_PATH = "data/sample_20.json"
OUTPUT_PATH = "data/llm_test_run.jsonl"
API_URL = "https://api.deepseek.com/v1/chat/completions"


def build_user_message(paper: dict) -> str:
    return (
        "<paper>\n"
        f"<paper_id>{paper.get('paper_id', '')}</paper_id>\n"
        f"<title>{paper.get('title', '')}</title>\n"
        f"<venue>{paper.get('venue', '')}</venue>\n"
        f"<year>{paper.get('year', '')}</year>\n"
        f"<abstract>{paper.get('abstract', '')}</abstract>\n"
        "</paper>"
    )


def call_deepseek(system: str, user: str, model: str) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set in environment")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    r = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-pro"

    system_prompt = Path(PROMPT_PATH).read_text(encoding="utf-8")
    sample_payload = json.loads(Path(SAMPLE_PATH).read_text(encoding="utf-8"))
    samples = sample_payload["samples"][:n]

    print(f"模型: {model}")
    print(f"待打分: {len(samples)} 篇")
    print(f"prompt 长度: {len(system_prompt)} 字符\n")

    results = []
    for i, paper in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {paper['paper_id']} | {paper['title'][:70]}", flush=True)
        t0 = time.time()
        record = {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "venue": paper.get("venue"),
            "year": paper.get("year"),
        }
        try:
            resp = call_deepseek(system_prompt, build_user_message(paper), model)
            msg = resp["choices"][0]["message"]
            usage = resp.get("usage", {})
            elapsed = time.time() - t0

            reasoning = msg.get("reasoning_content", "") or ""
            content = (msg.get("content", "") or "").strip()

            print(f"  耗时 {elapsed:.1f}s  reasoning={len(reasoning)}字  content={len(content)}字"
                  f"  tokens in/out={usage.get('prompt_tokens','?')}/{usage.get('completion_tokens','?')}",
                  flush=True)

            # 容错: 有些模型可能会用 ```json 围栏包裹
            stripped = content
            if stripped.startswith("```"):
                stripped = stripped.strip("`").lstrip("json").strip()

            try:
                parsed = json.loads(stripped)
                mr = parsed.get("mathematical_rigor", {}).get("score")
                tn = parsed.get("theoretical_novelty", {}).get("score")
                md = parsed.get("mathematical_depth", {}).get("score")
                conf = parsed.get("confidence_score", {}).get("score")
                dom = parsed.get("domain_modality")
                print(f"  ✓ JSON 合法  rigor={mr} novelty={tn} depth={md} conf={conf}  domain={dom!r}",
                      flush=True)
            except json.JSONDecodeError as e:
                parsed = None
                print(f"  ✗ JSON 解析失败: {e}", flush=True)
                print(f"  原始 content 前 200 字: {content[:200]}", flush=True)

            record.update({
                "elapsed_s": round(elapsed, 1),
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
                "reasoning_chars": len(reasoning),
                "reasoning": reasoning,
                "raw_content": content,
                "parsed": parsed,
                "ok": parsed is not None,
            })
        except requests.HTTPError as e:
            print(f"  HTTP 错误: {e}  body: {e.response.text[:300] if e.response else ''}",
                  flush=True)
            record.update({"ok": False, "error": str(e),
                           "error_body": e.response.text[:1000] if e.response else None})
        except Exception as e:
            print(f"  错误: {e}", flush=True)
            record.update({"ok": False, "error": repr(e)})

        results.append(record)
        time.sleep(0.3)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"\n成功 {ok_n}/{len(results)}  →  {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
