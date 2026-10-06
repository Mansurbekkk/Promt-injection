"""Main entry point: load a model and scan all apps.

Examples
--------
# Smoke test on CPU (no model), to confirm the pipeline:
python run.py --mock --out runs_mock.jsonl

# Real run on a GPU (Kaggle / Colab):
python run.py --model Qwen/Qwen2.5-7B-Instruct --out runs_qwen.jsonl

# Only Uzbek scripts, only two templates (faster):
python run.py --model Qwen/Qwen2.5-7B-Instruct --scripts uz-latn uz-cyrl \
              --templates context_ignoring fake_system --out runs_small.jsonl
"""
import argparse

from piscan.scanner import scan_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id")
    ap.add_argument("--mock", action="store_true", help="use scripted mock backend (CPU)")
    ap.add_argument("--out", default="runs.jsonl")
    ap.add_argument("--apps", nargs="*", default=None,
                    help="subset of: rag_qa email_agent web_summarizer")
    ap.add_argument("--scripts", nargs="*", default=None,
                    help="subset of: en uz-latn uz-cyrl mixed")
    ap.add_argument("--templates", nargs="*", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mock:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from tests.test_pipeline import mock_policy
        from piscan.backends import MockBackend
        backend = MockBackend(mock_policy)
    else:
        if not args.model:
            ap.error("give --model or --mock")
        from piscan.backends import HFBackend
        backend = HFBackend(args.model, max_new_tokens=args.max_new_tokens,
                            load_in_4bit=args.load_in_4bit)

    recs = scan_all(backend, args.out, apps=args.apps,
                    scripts=args.scripts, templates=args.templates, verbose=args.verbose)
    n_succ = sum(r["success"] for r in recs)
    n_comp = sum(r["competent"] for r in recs)
    print(f"\n{len(recs)} runs -> {args.out}")
    print(f"  successes: {n_succ}  competent: {n_comp}  ({n_comp/len(recs):.1%})")
    print(f"\nNow run:  python analyze.py {args.out}")


if __name__ == "__main__":
    main()
