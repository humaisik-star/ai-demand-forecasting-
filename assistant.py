"""
Demand & Stock Assistant - a natural-language layer over the forecasting and
inventory models, powered by Azure OpenAI function calling.

The LLM never sees the raw data; it answers by calling the tools in
src/assistant_tools.py (forecast lookup, stock recommendation, risk ranking,
portfolio summary) and explaining the results in plain language.

Setup (Azure OpenAI / AI Foundry):
    1. Create an Azure OpenAI resource and deploy a chat model (e.g. gpt-4o).
    2. Copy .env.example to .env and fill in your endpoint, key, and deployment.
    3. Make sure results/predictions.csv and results/stock_recommendations.csv
       exist (run predict.py and stock.py first).

Run:
    .venv/bin/python assistant.py                       # interactive chat
    .venv/bin/python assistant.py --ask "which products are most at risk?"
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.assistant_tools import TOOL_SPECS, dispatch

SYSTEM_PROMPT = """You are a demand-planning and inventory assistant for a retail chain.
Answer questions about product demand forecasts and stock recommendations by calling
the provided tools — never invent numbers. Store IDs look like S001..S005 and product
IDs like P0001..P0020. When the user is vague, call list_series or inventory_summary to
orient yourself. For CONCEPTUAL questions (definitions/why/how of ABC, ABC-XYZ, EOQ,
newsvendor, safety stock, reorder point, quantile forecasting, turnover, methodology),
call bilgi_ara and answer ONLY from the returned chunks, citing the source document
(e.g. "(kaynak: 03_eoq.md)"). If bilgi_ara returns found=false, OR the returned chunks do
not actually define the specific concept asked, say you don't know ("Bu konu bilgi
tabanımda yok.") — do not invent and do not stretch unrelated chunks. Keep answers
concise and business-focused: state the number, the unit,
and a one-line recommendation. If a tool returns an error, explain what's missing."""


def make_client():
    load_dotenv()
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print("Missing Azure OpenAI configuration:", ", ".join(missing))
        print("Copy .env.example to .env and fill it in (see the file header).")
        sys.exit(1)

    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    return client, os.environ["AZURE_OPENAI_DEPLOYMENT"]


def run_turn(client, deployment, messages, verbose=False):
    """Run one user turn, resolving any tool calls, and return the final text."""
    while True:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOL_SPECS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content

        # Record the assistant's tool-call request, then execute each call.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"  [tool] {tc.function.name}({args})")
            result = dispatch(tc.function.name, args)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, help="single question, then exit")
    parser.add_argument("--verbose", action="store_true", help="show tool calls")
    args = parser.parse_args()

    client, deployment = make_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if args.ask:
        messages.append({"role": "user", "content": args.ask})
        print(run_turn(client, deployment, messages, args.verbose))
        return

    print("Demand & Stock Assistant (Azure OpenAI). Type 'exit' to quit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit", ""}:
            break
        messages.append({"role": "user", "content": user})
        answer = run_turn(client, deployment, messages, args.verbose)
        print(f"\nassistant> {answer}\n")


if __name__ == "__main__":
    main()
