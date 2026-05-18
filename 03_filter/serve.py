#!/usr/bin/env python3
"""
论文分析本地服务器
用法: python3 serve.py [--port 8080]
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8080
PROJECT = Path(__file__).parent.parent

_candidate_full = PROJECT / "data/full/output.jsonl"
_candidate_flat = PROJECT / "data/output.jsonl"
DATA_FILE = _candidate_full if _candidate_full.exists() else _candidate_flat

HTML_FILE = Path(__file__).parent / "paper_analysis.html"
if not HTML_FILE.exists():
    HTML_FILE = Path.home() / "paper_analysis.html"

EXPLORE_DIR = PROJECT / "03_filter" / "rules"
FAV_FILE = PROJECT / "data/favorites.json"

PAPERS = []
PRESETS = {}     # name -> config dict
FAVORITES = set()  # paper_ids


# ── data loading ──────────────────────────────────────────────────────────────

def _avg(p):
    dims = ["mr", "tn", "md", "ar", "er", "cc", "ei", "sg"]
    vals = []
    for d in dims:
        v = p.get(d)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return round(sum(vals) / len(vals), 2) if vals else None


def load():
    print(f"Loading {DATA_FILE} …", flush=True)
    count = skip = 0
    with open(DATA_FILE) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("ok") or not d.get("parsed"):
                skip += 1
                continue
            p = d["parsed"]
            PAPERS.append({
                "paper_id": d["paper_id"],
                "title": d.get("title", ""),
                "venue": d.get("venue", ""),
                "year": d.get("year"),
                "parsed": p,
                "_avg": _avg(p),
            })
            count += 1
    print(f"Loaded {count} papers  (skipped {skip})", flush=True)

    # Load preset rule files
    preset_files = {
        "claude":     "rules_claude.json",
        "danchaofan": "rules_danchaofan.json",
        "jes":        "rules_jes.json",
    }
    for key, fname in preset_files.items():
        p = EXPLORE_DIR / fname
        if p.exists():
            PRESETS[key] = json.loads(p.read_text())
    print(f"Loaded {len(PRESETS)} rule presets: {list(PRESETS)}", flush=True)


def load_favorites():
    global FAVORITES
    if FAV_FILE.exists():
        try:
            FAVORITES = set(json.loads(FAV_FILE.read_text()))
        except Exception:
            FAVORITES = set()
    print(f"Loaded {len(FAVORITES)} favorites", flush=True)


def save_favorites():
    FAV_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAV_FILE.write_text(json.dumps(sorted(FAVORITES), ensure_ascii=False, indent=2))


# ── rules evaluation ──────────────────────────────────────────────────────────

def _field_val(parsed, field):
    if field == "marketing":
        return parsed.get("mk_f")
    if field == "human_review":
        return parsed.get("hr_f")
    if field == "integrity":
        return parsed.get("ig") or parsed.get("integrity")
    if field in ("mk_f", "hr_f"):
        return parsed.get(field)
    v = parsed.get(field)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


def eval_cond(parsed, c, keep_na=True):
    """
    keep_na=True  → NA 视作"不命中"（论文不会被这条 cond 剔除，倾向保留）
    keep_na=False → NA 视作"命中"（论文会被这条 cond 剔除，倾向排除）
    """
    field, op, value = c["field"], c["op"], c["value"]
    v = _field_val(parsed, field)
    if v is None:
        return not keep_na
    if op == "lt":  return v < value
    if op == "lte": return v <= value
    if op == "gt":  return v > value
    if op == "gte": return v >= value
    if op == "eq":  return v == value
    if op == "neq": return v != value
    if op == "in":
        lst = value if isinstance(value, list) else [value]
        return v in lst
    return False


def eval_rule(parsed, rule, keep_na=True):
    conds = rule.get("conditions", [])
    if not conds:
        return False
    results = [eval_cond(parsed, c, keep_na) for c in conds]
    logic = rule.get("internal_logic", "AND")
    hit = all(results) if logic == "AND" else any(results)
    return not hit if rule.get("negate") else hit


def eval_config(parsed, config):
    """Returns True if the paper is REJECTED (熔断) by the config."""
    active = [r for r in config.get("rules", []) if r.get("enabled", True)]
    if not active:
        return False
    keep_na = config.get("keep_na", True)
    hits = [eval_rule(parsed, r, keep_na) for r in active]
    logic = config.get("inter_logic", "OR")
    rejected = all(hits) if logic == "AND" else any(hits)
    if config.get("force_keep_hr") and parsed.get("hr_f"):
        rejected = False
    if rejected and config.get("rescue_rules"):
        for r in config["rescue_rules"]:
            if not r.get("enabled", True):
                continue
            if eval_rule(parsed, r, keep_na):
                rejected = False
                break
    return rejected


# ── simple filter + sort ──────────────────────────────────────────────────────

def filter_sort(q, venue, integrity, year, min_mr, max_er, sort_by):
    result = PAPERS
    if q:
        ql = q.lower()
        result = [p for p in result if ql in p["title"].lower()
                  or ql in (p["parsed"].get("dm") or "").lower()]
    if venue:
        result = [p for p in result if p["venue"] == venue]
    if integrity:
        result = [p for p in result if p["parsed"].get("ig") == integrity]
    if year:
        result = [p for p in result if str(p.get("year") or "") == year]
    if min_mr is not None:
        result = [p for p in result if (p["parsed"].get("mr") or 0) >= min_mr]
    if max_er is not None:
        result = [p for p in result if (p["parsed"].get("er") or 10) <= max_er]

    if sort_by == "avg_desc":
        result = sorted(result, key=lambda p: p["_avg"] or 0, reverse=True)
    elif sort_by == "avg_asc":
        result = sorted(result, key=lambda p: p["_avg"] or 0)
    elif sort_by == "mr_desc":
        result = sorted(result, key=lambda p: float(p["parsed"].get("mr") or 0), reverse=True)
    elif sort_by == "mr_asc":
        result = sorted(result, key=lambda p: float(p["parsed"].get("mr") or 0))
    elif sort_by == "er_asc":
        result = sorted(result, key=lambda p: float(p["parsed"].get("er") or 10))
    return result


def serialize(paper):
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "venue": paper["venue"],
        "year": paper["year"],
        "parsed": paper["parsed"],
        "is_favorite": paper["paper_id"] in FAVORITES,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        def qs1(key, default=None):
            v = qs.get(key)
            return v[0] if v else default

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML_FILE.read_bytes())

        elif path == "/api/papers":
            q        = qs1("q", "")
            venue    = qs1("venue", "")
            integrity= qs1("integrity", "")
            year     = qs1("year", "")
            sort_by  = qs1("sort", "")
            min_mr   = float(qs1("min_mr")) if qs.get("min_mr") else None
            max_er   = float(qs1("max_er")) if qs.get("max_er") else None
            page     = int(qs1("page", "1"))
            per_page = int(qs1("per_page", "50"))
            rules_json = qs1("rules", "")
            mode     = qs1("mode", "keep")  # keep | reject
            fav_only = qs1("fav_only", "")  # "1" → 只看收藏

            filtered = filter_sort(q, venue, integrity, year, min_mr, max_er, sort_by)

            if rules_json:
                try:
                    cfg = json.loads(rules_json)
                    result = []
                    for paper in filtered:
                        rejected = eval_config(paper["parsed"], cfg)
                        if (mode == "keep" and not rejected) or (mode == "reject" and rejected):
                            result.append(paper)
                    filtered = result
                except Exception:
                    pass  # bad rules JSON → ignore

            if fav_only == "1":
                filtered = [p for p in filtered if p["paper_id"] in FAVORITES]

            total = len(filtered)
            start = (page - 1) * per_page
            body = json.dumps({
                "total": total,
                "page": page,
                "per_page": per_page,
                "papers": [serialize(p) for p in filtered[start: start + per_page]],
            }, ensure_ascii=False).encode()
            self._send(200, "application/json", body)

        elif path == "/api/venues":
            venues = sorted({p["venue"] for p in PAPERS if p["venue"]})
            self._send(200, "application/json", json.dumps(venues).encode())

        elif path == "/api/years":
            years = sorted({str(p["year"]) for p in PAPERS if p.get("year")}, reverse=True)
            self._send(200, "application/json", json.dumps(years).encode())

        elif path == "/api/presets":
            self._send(200, "application/json",
                       json.dumps(PRESETS, ensure_ascii=False).encode())

        elif path == "/api/favorites":
            self._send(200, "application/json",
                       json.dumps(sorted(FAVORITES)).encode())

        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/favorites/toggle":
            pid = (qs.get("paper_id") or [None])[0]
            if not pid:
                self._send(400, "application/json",
                           json.dumps({"error": "missing paper_id"}).encode())
                return
            if pid in FAVORITES:
                FAVORITES.discard(pid)
                is_fav = False
            else:
                FAVORITES.add(pid)
                is_fav = True
            save_favorites()
            self._send(200, "application/json",
                       json.dumps({"paper_id": pid, "is_favorite": is_fav,
                                   "total": len(FAVORITES)}).encode())
        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    load()
    load_favorites()
    print(f"Serving at http://localhost:{args.port}", flush=True)
    HTTPServer(("localhost", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
