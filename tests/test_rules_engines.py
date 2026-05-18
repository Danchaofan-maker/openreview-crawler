import importlib.util
import json
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/full/output.jsonl"
RULE_PATHS = [
    ROOT / "03_filter/rules/rules_jes.json",
    ROOT / "03_filter/rules/rules_danchaofan.json",
    ROOT / "03_filter/rules/rules_claude.json",
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MMR = load_module("mmr_select", ROOT / "03_filter/mmr_select.py")
SERVE = load_module("serve", ROOT / "03_filter/serve.py")


def load_papers(path):
    papers = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("ok") and row.get("parsed"):
                parsed = dict(row["parsed"])
                parsed["_test_pid"] = row.get("paper_id") or parsed.get("id")
                papers.append(parsed)
    return papers


def kept_ids_mmr(papers, config):
    return {p["_test_pid"] for p in MMR.apply_rules(papers, config)}


def kept_ids_serve(papers, config):
    return {p["_test_pid"] for p in papers if not SERVE.eval_config(p, config)}


def rejected_ids_serve(papers, config):
    return {p["_test_pid"] for p in papers if SERVE.eval_config(p, config)}


class RulesEngineConsistencyTest(unittest.TestCase):
    def assert_engines_equal(self, papers, config):
        self.assertEqual(kept_ids_mmr(papers, config), kept_ids_serve(papers, config))

    def test_object_engine_edge_cases_match(self):
        papers = [
            {"_test_pid": "low", "mr": 2, "er": 6, "tea": None, "hr_f": False, "ig": "intact"},
            {"_test_pid": "high", "mr": 8, "er": 9, "tea": 5, "hr_f": False, "ig": "intact"},
            {"_test_pid": "hr", "mr": 1, "er": 9, "tea": 4, "hr_f": True, "ig": "absent"},
        ]
        configs = [
            {
                "rules": [{"conditions": [{"field": "mr", "op": "lte", "value": 3}, {"field": "er", "op": "gte", "value": 5}]}],
            },
            {
                "rules": [{"internal_logic": "AND", "conditions": [{"field": "er", "op": "gt", "value": 4}, {"field": "tea", "op": "lt", "value": 6.5}]}],
            },
            {
                "rules": [{"internal_logic": "OR", "conditions": [{"field": "tea", "op": "lt", "value": 6.5}, {"field": "er", "op": "gt", "value": 4}]}],
            },
            {
                "rules": [{"enabled": False, "conditions": [{"field": "mr", "op": "lte", "value": 3}]}],
            },
            {
                "rules": [{"negate": True, "conditions": [{"field": "mr", "op": "gte", "value": 7}]}],
            },
            {
                "rules": [{"conditions": []}],
            },
            {
                "inter_logic": "AND",
                "rules": [
                    {"conditions": [{"field": "mr", "op": "gte", "value": 7}]},
                    {"conditions": [{"field": "er", "op": "gte", "value": 8}]},
                ],
            },
            {
                "keep_na": False,
                "rules": [{"conditions": [{"field": "tea", "op": "lt", "value": 6.5}]}],
            },
            {
                "force_keep_hr": True,
                "rules": [{"conditions": [{"field": "integrity", "op": "eq", "value": "absent"}]}],
            },
            {
                "rules": [{"conditions": [{"field": "mr", "op": "gte", "value": 7}]}],
                "rescue_rules": [{"enabled": False, "conditions": [{"field": "mr", "op": "gte", "value": 7}]}],
            },
            {
                "rules": [{"conditions": [{"field": "mr", "op": "gte", "value": 7}]}],
                "rescue_rules": [{"conditions": [{"field": "mr", "op": "gte", "value": 7}]}],
            },
        ]
        for config in configs:
            with self.subTest(config=config):
                self.assert_engines_equal(papers, config)

    def test_metamorphic_rule_order_and_disabled_rules(self):
        papers = [
            {"_test_pid": "a", "mr": 2, "er": 2, "hr_f": False, "ig": "intact"},
            {"_test_pid": "b", "mr": 8, "er": 2, "hr_f": False, "ig": "partial"},
            {"_test_pid": "c", "mr": 8, "er": 9, "hr_f": False, "ig": "intact"},
            {"_test_pid": "d", "mr": 4, "er": 9, "hr_f": True, "ig": "absent"},
        ]
        config = {
            "inter_logic": "OR",
            "rules": [
                {"conditions": [{"field": "mr", "op": "lte", "value": 3}]},
                {"conditions": [{"field": "er", "op": "gte", "value": 8}]},
                {"conditions": [{"field": "integrity", "op": "eq", "value": "partial"}]},
            ],
        }
        base = kept_ids_mmr(papers, config)

        reordered = {**config, "rules": list(reversed(config["rules"]))}
        self.assertEqual(base, kept_ids_mmr(papers, reordered))

        with_disabled = {
            **config,
            "rules": config["rules"] + [
                {"enabled": False, "conditions": [{"field": "mr", "op": "gte", "value": 0}]}
            ],
        }
        self.assertEqual(base, kept_ids_mmr(papers, with_disabled))

        and_config = {
            "inter_logic": "AND",
            "rules": [
                {"conditions": [{"field": "mr", "op": "gte", "value": 7}]},
                {"conditions": [{"field": "er", "op": "gte", "value": 8}]},
            ],
        }
        and_reordered = {**and_config, "rules": list(reversed(and_config["rules"]))}
        self.assertEqual(kept_ids_mmr(papers, and_config), kept_ids_mmr(papers, and_reordered))

    def test_metamorphic_condition_order_rescue_and_force_keep(self):
        papers = [
            {"_test_pid": "low", "mr": 2, "er": 9, "hr_f": False, "ig": "intact"},
            {"_test_pid": "rescued", "mr": 8, "er": 9, "hr_f": False, "ig": "intact"},
            {"_test_pid": "hr", "mr": 1, "er": 9, "hr_f": True, "ig": "absent"},
            {"_test_pid": "miss", "mr": 6, "er": None, "hr_f": False, "ig": "partial"},
        ]
        rule = {
            "internal_logic": "AND",
            "conditions": [
                {"field": "mr", "op": "gte", "value": 7},
                {"field": "er", "op": "gte", "value": 8},
            ],
        }
        swapped = {**rule, "conditions": list(reversed(rule["conditions"]))}
        self.assertEqual(MMR._eval_rule(papers[1], rule), MMR._eval_rule(papers[1], swapped))
        self.assertEqual(SERVE.eval_rule(papers[1], rule), SERVE.eval_rule(papers[1], swapped))

        rejecting = {"rules": [{"conditions": [{"field": "er", "op": "gte", "value": 8}]}]}
        rescued = {
            **rejecting,
            "rescue_rules": [{"conditions": [{"field": "mr", "op": "gte", "value": 7}]}],
        }
        self.assertIn("rescued", rejected_ids_serve(papers, rejecting))
        self.assertIn("rescued", kept_ids_serve(papers, rescued))

        force_keep = {
            "force_keep_hr": True,
            "rules": [{"conditions": [{"field": "integrity", "op": "eq", "value": "absent"}]}],
        }
        self.assertIn("hr", kept_ids_mmr(papers, force_keep))
        self.assertIn("hr", kept_ids_serve(papers, force_keep))

    def test_randomized_object_engines_match(self):
        rng = random.Random(20260518)
        numeric_fields = ["mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs"]
        bool_fields = ["mk_f", "hr_f", "marketing", "human_review"]
        fields = numeric_fields + bool_fields + ["integrity"]

        papers = []
        for idx in range(60):
            paper = {"_test_pid": f"p{idx}"}
            for field in numeric_fields:
                if rng.random() < 0.2:
                    paper[field] = None
                elif rng.random() < 0.15:
                    continue
                else:
                    paper[field] = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            paper["mk_f"] = rng.choice([True, False])
            paper["hr_f"] = rng.choice([True, False])
            if rng.random() < 0.9:
                paper["ig"] = rng.choice(["intact", "partial", "broken", "absent"])
            papers.append(paper)

        def make_condition():
            field = rng.choice(fields)
            if field in numeric_fields:
                return {
                    "field": field,
                    "op": rng.choice(["lt", "lte", "gt", "gte", "eq", "neq"]),
                    "value": rng.choice([0, 2, 4, 6, 8, 10]),
                }
            if field in bool_fields:
                return {"field": field, "op": "eq", "value": rng.choice([True, False])}
            return {
                "field": "integrity",
                "op": rng.choice(["eq", "neq", "in"]),
                "value": rng.choice(
                    ["intact", "partial", ["broken", "absent"], ["intact", "partial"]]
                ),
            }

        def make_rule():
            return {
                "enabled": rng.random() > 0.15,
                "negate": rng.random() < 0.15,
                "internal_logic": rng.choice(["AND", "OR"]),
                "conditions": [make_condition() for _ in range(rng.randrange(0, 4))],
            }

        for idx in range(200):
            config = {
                "inter_logic": rng.choice(["AND", "OR"]),
                "force_keep_hr": rng.choice([True, False]),
                "keep_na": rng.choice([True, False]),
                "rules": [make_rule() for _ in range(rng.randrange(0, 5))],
                "rescue_rules": [make_rule() for _ in range(rng.randrange(0, 3))],
            }
            with self.subTest(random_config=idx):
                self.assert_engines_equal(papers, config)

    @unittest.skipUnless(DATA_PATH.exists(), "full data file is not available")
    def test_object_engines_match_on_rule_files(self):
        papers = load_papers(DATA_PATH)
        for path in RULE_PATHS:
            with self.subTest(rules=path.name):
                config = json.loads(path.read_text())
                self.assert_engines_equal(papers, config)

    @unittest.skipUnless(DATA_PATH.exists(), "full data file is not available")
    def test_rules_jes_keeps_6236_papers(self):
        papers = load_papers(DATA_PATH)
        config = json.loads((ROOT / "03_filter/rules/rules_jes.json").read_text())
        self.assertEqual(len(kept_ids_mmr(papers, config)), 6236)
        self.assertEqual(len(kept_ids_serve(papers, config)), 6236)


if __name__ == "__main__":
    unittest.main()
