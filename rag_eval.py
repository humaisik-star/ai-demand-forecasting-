"""
RAG evaluation — a 10-question test set that measures the assistant's grounded
answering accuracy end-to-end (the LLM decides to call `bilgi_ara`, answers from
retrieved chunks, cites the source, and abstains when the topic is out of scope).

Scoring per question:
  * in-domain: correct if bilgi_ara was used AND (expected source cited OR an
    expected keyword appears in the answer).
  * out-of-domain: correct if the assistant abstains ("bilmiyorum" / "bilgi yok").

Run:
    .venv/bin/python rag_eval.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from assistant import make_client, run_turn, SYSTEM_PROMPT

EVAL = [
    {"q": "ABC analizi nedir?", "source": "02_abc_analizi.md", "kw": ["pareto", "80"]},
    {"q": "EOQ formülü nedir ve ne işe yarar?", "source": "03_eoq.md", "kw": ["sipariş", "√", "2"]},
    {"q": "Güvenlik stoğu nasıl hesaplanır?", "source": "04_guvenlik_stogu.md", "kw": ["z", "servis"]},
    {"q": "Yeniden sipariş noktası (reorder point) nedir?", "source": "05_reorder_point.md", "kw": ["tedarik", "lead"]},
    {"q": "Newsvendor modeli ve kritik oran nedir?", "source": "06_newsvendor.md", "kw": ["kritik", "Cu"]},
    {"q": "Quantile tahmin ve P90 ne anlama gelir?", "source": "07_quantile_tahmin.md", "kw": ["P90", "aralık"]},
    {"q": "ABC-XYZ analizinde XYZ neyi ölçer?", "source": "08_abc_xyz.md", "kw": ["değişkenlik", "cv"]},
    {"q": "Stok devir hızı nedir?", "source": "09_stok_devir_ve_tahmin.md", "kw": ["envanter", "yıl"]},
    # out-of-domain (inventory-adjacent concepts NOT in the KB) -> must abstain
    {"q": "Just-in-time (JIT) envanter sistemi nedir?", "source": None, "kw": []},
    {"q": "Kanban stok yönetimi nasıl çalışır?", "source": None, "kw": []},
]

ABSTAIN = ["bilmiyorum", "bilgi yok", "bilgi tabanım", "bulunmuyor", "yer almıyor",
           "sahip değil", "tabanımda yok", "kapsam", "içermiyor", "tanımlı değil", "mevcut değil"]


def score(row, answer, tools):
    a = answer.lower()
    if row["source"] is None:
        return any(x in a for x in ABSTAIN)          # should abstain
    used = "bilgi_ara" in tools
    cited = row["source"].lower() in a
    kw = any(k.lower() in a for k in row["kw"])
    return used and (cited or kw)


def main():
    load_dotenv()
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("Missing Azure OpenAI config."); sys.exit(1)
    client, deployment = make_client()

    results, correct = [], 0
    for row in EVAL:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["q"]}]
        tools = []
        # capture tool calls by wrapping run_turn's dispatch via messages inspection
        answer = run_turn(client, deployment, messages)
        tools = [m.get("tool_calls", [{}])[0].get("function", {}).get("name")
                 for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        ok = score(row, answer or "", tools)
        correct += ok
        results.append({"soru": row["q"], "beklenen_kaynak": row["source"] or "(abstain)",
                        "tools": ",".join(t for t in tools if t), "dogru": ok,
                        "cevap": (answer or "")[:120]})
        print(f"{'✅' if ok else '❌'}  {row['q']}")
        print(f"     tools={results[-1]['tools']}  →  {results[-1]['cevap']}")

    acc = correct / len(EVAL)
    print(f"\n=== RAG doğruluk: {correct}/{len(EVAL)} = %{acc*100:.0f} ===")

    Path("results").mkdir(exist_ok=True)
    import csv
    with open("results/rag_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print("Saved -> results/rag_eval.csv")


if __name__ == "__main__":
    main()
