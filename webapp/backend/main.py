"""FastAPI backend for the Demand & Stock Assistant.

Exposes a /chat endpoint that runs the Azure OpenAI function-calling loop over
the tools in tools.py. CORS is open so a Vercel-hosted frontend can call it.

Env vars (set in Azure Container Apps / locally in .env):
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_API_VERSION (optional)
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import pandas as pd

from store import load as _load_table
from tools import DATA_DIR, TOOL_SPECS, dispatch

STATIC_DIR = Path(__file__).parent / "static"


def _csv(name):
    """Read an analysis table from analysis.db when present, else the CSV."""
    table = name[:-4] if name.endswith(".csv") else name
    return _load_table(table, DATA_DIR / name)

SYSTEM_PROMPT = """You are a demand-planning and inventory assistant for a retail chain.
Answer by calling the tools — never invent numbers. Store IDs look like S001..S005 and
product IDs like P0001..P0020. When the user is vague, call list_series or
inventory_summary to orient yourself. You can do demand forecasts, stock recommendations,
ABC / ABC-XYZ analysis, EOQ and newsvendor order sizing, reorder points, z-score safety
stock, turnover, and stockout/anomaly alerts. For a "yönetici özeti" call yonetici_ozeti.

For CONCEPTUAL questions (what is / why / how — ABC, ABC-XYZ, EOQ, newsvendor, safety
stock, reorder point, quantile forecasting, turnover, methodology) call bilgi_ara and
answer ONLY from the returned chunks. Do NOT add a "Kaynak" line — the interface already
shows the source. If bilgi_ara returns found=false or the chunks do not actually define
what was asked, reply exactly "Bu konu bilgi tabanımda yok." — never invent or stretch.

HOW TO WRITE THE ANSWER:
- Match length to the question. A simple question gets one or two sentences. A complex
  question gets a fuller, structured answer. Never force every reply into the same shape.
- Write plain, direct sentences. Do not put extra explanations in parentheses.
- Use Markdown when it helps clarity: a short bold lead-in or a `##` heading for a longer
  answer, **bold for the key numbers**, short bullet lists for several items, and a table
  when comparing rows. Do not over-format a short answer — a one-line answer needs no
  heading or list.
- Format numbers Turkish-style: dot as the thousands separator (**15.651**), comma for
  decimals (**94,7**), and keep the percent sign attached (**%28,4**). Be consistent.
- Be efficient: answer greetings and simple clarifications directly without any tool.
  Call only the tools you actually need and never repeat a tool call you already made.
Reply in the user's language (Turkish by default)."""

MAX_TOKENS = 1400  # cap answer length to keep responses fast and focused

app = FastAPI(title="Demand & Stock Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None


def client():
    """Lazily build the Azure OpenAI client so /health works without creds."""
    global _client
    if _client is None:
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return _client


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str] = []
    chart: dict | None = None
    sources: list[dict] = []
    followups: list[str] = []


# Friendly labels for the RAG source badge.
DOC_LABELS = {
    "01_proje_metodoloji.md": "Metodoloji",
    "02_abc_analizi.md": "ABC Analizi",
    "03_eoq.md": "EOQ",
    "04_guvenlik_stogu.md": "Güvenlik Stoğu",
    "05_reorder_point.md": "Yeniden Sipariş Noktası",
    "06_newsvendor.md": "Newsvendor",
    "07_quantile_tahmin.md": "Quantile Tahmin",
    "08_abc_xyz.md": "ABC-XYZ",
    "09_stok_devir_ve_tahmin.md": "Stok Devir & Tahmin",
}

# Context-aware follow-up suggestions, keyed by the primary tool used.
FOLLOWUPS = {
    "bilgi_ara": ["Güvenlik stoğu nasıl hesaplanır?", "Newsvendor modeli nedir?", "ABC-XYZ neyi ölçer?"],
    "get_demand_forecast": ["Bu ürün için stok önerisi ver", "En riskli ürünler hangileri?", "ABC sınıfı nedir?"],
    "get_stock_recommendation": ["Bu ürünün EOQ değeri nedir?", "Newsvendor sipariş miktarı nedir?", "Yönetici özeti ver"],
    "get_inventory_policy": ["Bu ürünün talep trendi nasıl?", "En riskli ürünler hangileri?", "Newsvendor nedir?"],
    "get_advanced_policy": ["ABC-XYZ neyi ölçer?", "Newsvendor modeli nedir?", "Stok devir hızı nedir?"],
    "list_stockout_alerts": ["Yönetici özeti ver", "ABC analizini özetle", "Anomalileri açıkla"],
    "abc_summary": ["ABC-XYZ analizini göster", "En değerli ürünler hangileri?", "EOQ nedir?"],
    "list_top_stockout_risks": ["Yönetici özeti ver", "Reorder gereken ürünler", "Güvenlik stoğu nedir?"],
    "yonetici_ozeti": ["En riskli ürünleri listele", "Anomalileri açıkla", "ABC-XYZ analizini göster"],
    "inventory_summary": ["ABC analizini özetle", "Reorder gereken ürünler", "Yönetici özeti ver"],
}
DEFAULT_FOLLOWUPS = ["Yönetici özeti ver", "ABC analizini özetle", "EOQ nedir?"]


def followups_for(tools_used):
    """2-3 suggestions based on the first recognised tool used."""
    for t in tools_used:
        if t in FOLLOWUPS:
            return FOLLOWUPS[t][:3]
    return DEFAULT_FOLLOWUPS


def _chart_from(name, args, result):
    """Build inline chart data when a tool result is chartable (demand trend)."""
    if name == "get_demand_forecast" and isinstance(result, dict) and result.get("rows"):
        rows = result["rows"]
        return {
            "type": "line",
            "title": f"{args.get('store_id')}/{args.get('product_id')} — talep trendi",
            "labels": [r["Date"][5:] for r in rows],  # MM-DD
            "series": [
                {"name": "Tahmin", "values": [r.get("Predicted_Demand") for r in rows]},
                {"name": "Gerçek", "values": [r.get("Actual_Demand") for r in rows]},
            ],
        }
    return None


def run_turn(messages):
    """Resolve tool calls until the model returns a text answer."""
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    tools_used = []
    chart = None
    sources = []
    while True:
        resp = client().chat.completions.create(
            model=deployment, messages=messages, tools=TOOL_SPECS, tool_choice="auto",
            max_completion_tokens=MAX_TOKENS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, tools_used, chart, sources

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tools_used.append(tc.function.name)
            result = dispatch(tc.function.name, args)
            chart = _chart_from(tc.function.name, args, result) or chart
            if tc.function.name == "bilgi_ara" and isinstance(result, dict) and result.get("found"):
                seen = {s["file"] for s in sources}
                for s in result.get("sources", []):
                    if s["source"] not in seen:
                        seen.add(s["source"])
                        sources.append({"file": s["source"],
                                        "label": DOC_LABELS.get(s["source"], s["source"])})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Demand & Stock Assistant API. POST /chat to talk."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/kpis")
def api_kpis():
    """Hero KPIs for the dashboard."""
    inv = _csv("inventory_analytics.csv")
    stock = _csv("stock_recommendations.csv")
    naive = float(stock["avg_inventory_naive"].sum())
    model = float(stock["avg_inventory_model"].sum())
    try:
        r2 = float(_csv("model_metrics.csv").sort_values("RMSE").iloc[0]["R2"])
    except Exception:
        r2 = None
    return {
        "total_skus": int(len(inv)),
        "model_r2": round(r2, 3) if r2 is not None else None,
        "inventory_reduction_pct": round((naive - model) / naive * 100, 1),
        "avg_service_level_pct": round(float(stock["service_model"].mean()) * 100, 1),
        "reorder_alerts": int((inv["alert_status"] != "OK").sum()),
        "critical_alerts": int((inv["alert_status"] == "CRITICAL").sum()),
        "class_a_skus": int((inv["abc_class"] == "A").sum()),
        "abc": {c: int((inv["abc_class"] == c).sum()) for c in ["A", "B", "C"]},
        "alert_counts": {
            s: int((inv["alert_status"] == s).sum()) for s in ["CRITICAL", "REORDER", "OK"]
        },
    }


@app.get("/api/inventory")
def api_inventory():
    """Full per-SKU table for the filterable product grid."""
    inv = _csv("inventory_analytics.csv")
    return {"rows": inv.to_dict("records")}


@app.get("/api/metrics")
def api_metrics():
    """Model comparison / backtest / quantile tables for the Reports tab."""
    out = {}
    for key, fname in [
        ("model", "model_metrics.csv"),
        ("backtest", "backtest_metrics.csv"),
        ("quantile", "quantile_metrics.csv"),
    ]:
        try:
            out[key] = _csv(fname).to_dict("records")
        except Exception:
            out[key] = []
    return out


@app.get("/api/advanced")
def api_advanced():
    """ABC-XYZ matrix, anomalies, and advanced per-SKU policy for the Reports tab."""
    try:
        matrix = _csv("abc_xyz_matrix.csv")
        matrix_rows = matrix.rename(columns={matrix.columns[0]: "ABC"}).to_dict("records")
    except Exception:
        matrix_rows = []
    try:
        anomalies = _csv("anomalies.csv").to_dict("records")
    except Exception:
        anomalies = []
    try:
        adv = _csv("advanced_analytics.csv").to_dict("records")
    except Exception:
        adv = []
    return {"matrix": matrix_rows, "anomalies": anomalies, "advanced": adv}


@app.exception_handler(Exception)
async def json_error_handler(request: Request, exc: Exception):
    """Return JSON (not a raw HTML 500) so the frontend can always parse it."""
    return JSONResponse(
        status_code=200,
        content={
            "answer": "Şu an bir sorun oluştu, lütfen birkaç saniye sonra tekrar deneyin.",
            "tools_used": [],
            "error": str(exc),
        },
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": t.role, "content": t.content} for t in req.history]
        messages.append({"role": "user", "content": req.message})
        answer, tools_used, chart, sources = run_turn(messages)
        # Only show source badges when the answer actually used the knowledge base.
        show_sources = sources[:2] if (answer and "bilgi tabanımda yok" not in answer) else []
        return ChatResponse(
            answer=answer or "", tools_used=tools_used, chart=chart,
            sources=show_sources, followups=followups_for(tools_used),
        )
    except Exception:
        # Never surface a raw 500; hand the frontend a clean, polite JSON message.
        return ChatResponse(
            answer="Şu an bir aksaklık oldu, lütfen birkaç saniye sonra tekrar deneyin.",
            tools_used=[],
        )


def _capture_sources(name, result, sources):
    if name == "bilgi_ara" and isinstance(result, dict) and result.get("found"):
        seen = {s["file"] for s in sources}
        for s in result.get("sources", []):
            if s["source"] not in seen:
                seen.add(s["source"])
                sources.append({"file": s["source"], "label": DOC_LABELS.get(s["source"], s["source"])})


def stream_turn(messages):
    """Yield ('delta', text) for answer tokens and ('meta', {...}) at the end.

    Tool rounds are consumed silently (accumulate tool-call deltas, then execute);
    only the final assistant text is streamed token by token to the client.
    """
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    tools_used, sources, chart = [], [], None
    while True:
        stream = client().chat.completions.create(
            model=deployment, messages=messages, tools=TOOL_SPECS, tool_choice="auto",
            max_completion_tokens=MAX_TOKENS, stream=True,
        )
        content, calls = [], {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content.append(delta.content)
                yield ("delta", delta.content)
            if delta and delta.tool_calls:
                for tcd in delta.tool_calls:
                    c = calls.setdefault(tcd.index, {"id": None, "name": "", "args": ""})
                    if tcd.id:
                        c["id"] = tcd.id
                    if tcd.function and tcd.function.name:
                        c["name"] = tcd.function.name
                    if tcd.function and tcd.function.arguments:
                        c["args"] += tcd.function.arguments
        if not calls:
            answer = "".join(content)
            show = sources[:2] if "bilgi tabanımda yok" not in answer else []
            yield ("meta", {"tools_used": tools_used, "chart": chart,
                            "sources": show, "followups": followups_for(tools_used)})
            return
        messages.append({
            "role": "assistant", "content": "".join(content) or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"]}}
                           for c in calls.values()],
        })
        for c in calls.values():
            args = json.loads(c["args"] or "{}")
            tools_used.append(c["name"])
            result = dispatch(c["name"], args)
            chart = _chart_from(c["name"], args, result) or chart
            _capture_sources(c["name"], result, sources)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-Sent Events: stream the answer token by token (stream=true)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": t.role, "content": t.content} for t in req.history]
    messages.append({"role": "user", "content": req.message})

    def gen():
        try:
            for kind, payload in stream_turn(messages):
                key = "delta" if kind == "delta" else "done"
                yield f"data: {json.dumps({key: payload}, ensure_ascii=False)}\n\n"
        except Exception:
            yield f"data: {json.dumps({'error': 'Şu an bir aksaklık oldu, lütfen tekrar deneyin.'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
