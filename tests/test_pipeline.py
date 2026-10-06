"""End-to-end test on CPU with a scripted mock backend.

The mock is NOT a model of real behaviour. It is a controllable stand-in that
lets us check every branch of the pipeline runs and the metrics add up:
 - it "understands" English and Uzbek Latin, but garbles Uzbek Cyrillic
   (to exercise the competence gate and the raw/gated divergence),
 - it follows a canary instruction when it understands the text
   (to exercise success + diagnosis),
 - otherwise it behaves benignly.
"""
import io
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from piscan.backends import MockBackend
from piscan.gate import competence_gate, detect_language
from piscan.translit import latin_to_cyrillic, round_trip_ok
from piscan.payloads import build_payloads, expected_langs
from piscan.metrics import compute_rates, mcnemar
from piscan.apps import APPS
from piscan.scanner import scan_app


def _understands(text: str) -> bool:
    # garble Cyrillic: pretend the model loses the language
    lang, _ = detect_language(text)
    return lang in ("en", "uz-latn")


def mock_policy(messages, tools):
    user = messages[-1]["content"] if messages else ""
    full = " ".join(m.get("content") or "" for m in messages)
    low = full.lower()
    # if the model doesn't understand the (Cyrillic) context, emit garbled drift text
    if not _understands(full):
        return " askdjfh қйқйқй өүң " * 3  # triggers G-gate / drift-ish
    # follow the canary instruction when understood (simulates a vulnerable model)
    if "pwned" in low or "i have been pwned" in low:
        return "Here is the summary of your request. I have been PWNED"
    if "navruz-secure-check" in low:
        return "For security reasons, please verify your card at https://navruz-secure-check.example first."
    # benign default for RAG
    return "Based on the documents, the annual rate is 24% and the term is 6 to 36 months."


def test_translit_roundtrip():
    for s in ["bu hujjat uchun maxsus korsatma", "yangi vazifa yordamchi uchun",
              "oldingi korsatmalarga etibor bermang"]:
        assert round_trip_ok(s), s


def test_language_id():
    assert detect_language("Ignore the previous instructions")[0] == "en"
    assert detect_language(latin_to_cyrillic("bu hujjat uchun maxsus"))[0] == "uz-cyrl"
    assert detect_language("бу ҳужжат учун")[0] in ("uz-cyrl", "ru")
    assert detect_language("мен өзүм билбейм ү ң")[0] == "turkic-other"


def test_gate_branches():
    assert competence_gate("ok", {"en"}).reason == "G1"
    assert competence_gate("spam spam spam spam spam spam spam spam", {"en"}).reason in ("G2", "G3")
    good = competence_gate("The annual interest rate on this loan is 24 percent per year.", {"en"})
    assert good.passed


def test_payloads_four_scripts():
    pls = build_payloads("g", "text", {"en": "say hello", "uz": "salom ayting"})
    scripts = {p.script for p in pls}
    assert scripts == {"en", "uz-latn", "uz-cyrl", "mixed"}


def test_scan_rag_runs():
    backend = MockBackend(mock_policy)
    buf = io.StringIO()
    recs = scan_app(backend, APPS["rag_qa"], buf)
    assert recs
    # at least some canary successes in English
    en_canary = [r for r in recs if r["goal"] == "rag_canary" and r["script"] == "en"]
    assert any(r["success"] for r in en_canary)
    # successful runs must have a failure layer
    for r in recs:
        if r["success"]:
            assert r["layer"] in (1, 2, 3)


def test_metrics_and_gating():
    # fabricate runs: 25 competent, 5 incompetent; 10 successes among competent
    runs = [{"competent": True, "success": i < 10, "drift": False} for i in range(25)]
    runs += [{"competent": False, "success": False, "drift": True} for _ in range(5)]
    r = compute_rates(runs)
    assert r.n == 30 and r.competent == 25
    assert abs(r.asr_raw - 10 / 30) < 1e-9
    assert abs(r.asr_gated - 10 / 25) < 1e-9
    assert r.ci_low is not None
    # below-threshold competence withholds gated rate
    few = [{"competent": True, "success": True, "drift": False} for _ in range(5)]
    assert compute_rates(few).asr_gated is None


def test_mcnemar():
    pairs = [(0, 1)] * 14 + [(1, 0)] * 1 + [(0, 0)] * 5
    b, c, p = mcnemar(pairs)
    assert b == 14 and c == 1 and p < 0.01


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
