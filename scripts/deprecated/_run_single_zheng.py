#!/usr/bin/env python3
"""临时单篇运行脚本，逻辑与 score_paper_test_v4pro_thinking.py 完全一致"""
import sys
sys.path.insert(0, "scripts")
import score_paper_test_v4pro_thinking as base

base.SAMPLE_PATH = "data/sample_temp_zheng.json"
base.OUTPUT_PATH = "data/llm_test_zheng.jsonl"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "1"]   # N=1，只跑这一篇
    base.main()
