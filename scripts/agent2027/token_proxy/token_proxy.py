#!/usr/bin/env python3
"""X5 · token-logging proxy for the external arms (01 §3.3 X5; A08-T4/T12, A21, A23).

A minimal OpenAI-compatible pass-through that sits between an external client (LLM-FE) and
the session's vLLM server. It changes nothing in the request or the response body; it only

* swaps the client's bearer key for the server key (the client key is the ATTRIBUTION
  label, e.g. ``a08-llmfe-synp_p2-r0``; the server key never reaches the client), and
* records, per request, the client label and the ``usage`` block vLLM returns
  (prompt / completion tokens), plus status, latency and finish reasons.

Outputs (``--log-dir``):
  ``requests.jsonl``  one JSON line per proxied request (append-only);
  ``totals.json``     per-client totals, rewritten after every request (atomic rename).
On SIGTERM / SIGINT it writes ``totals.json`` once more and, with ``--wandb``, logs one W&B
run ``<SESSION_ID>__proxy`` (group ``A08`` by default, job_type ``token_proxy``) holding the
per-client totals table, ``tokens/<client>/{prompt,completion,calls}`` summary keys and the
two files as artifact ``proxy-log``. W&B failures never stop the proxy.

``GET /proxy/totals`` returns the live totals (served by the proxy, never forwarded).
Everything else (``/health``, ``/v1/models``, ``/v1/chat/completions``, ``/v1/completions``)
is forwarded verbatim. Streaming responses are relayed chunk by chunk; their tokens are
counted only if the stream carries a ``usage`` object (vLLM: ``stream_options.include_usage``),
otherwise the request is flagged ``usage_missing``.

Standard library only (runs in any of our images). Run on CAIR in a container, e.g.::

  docker run -d --name ro27-<sid>-proxy --network host -e UPSTREAM_API_KEY -e WANDB_API_KEY \\
    -v <P>/scripts/agent2027/token_proxy:/proxy:ro -v ~/ro27_a08/proxy/<SID>:/out \\
    --entrypoint python3 rogue_one:agent2027-v2.N /proxy/token_proxy.py \\
    --listen-port 8090 --upstream http://127.0.0.1:8020 --log-dir /out \\
    --session-id <SID> --wandb
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "content-length", "host"}


class Ledger:
    """Thread-safe per-client token ledger with a JSONL request log."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.totals: dict[str, dict] = {}
        self.req_path = self.log_dir / "requests.jsonl"
        self.tot_path = self.log_dir / "totals.json"

    def record(self, rec: dict) -> None:
        with self.lock:
            t = self.totals.setdefault(rec["client"], {
                "calls": 0, "ok_calls": 0, "error_calls": 0, "http_400": 0, "finish_length": 0,
                "usage_missing": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "latency_s": 0.0, "first_ts": rec["ts"], "last_ts": rec["ts"]})
            t["calls"] += 1
            ok = 200 <= int(rec["status"]) < 300
            t["ok_calls" if ok else "error_calls"] += 1
            t["http_400"] += int(rec["status"]) == 400
            t["finish_length"] += sum(1 for f in rec.get("finish_reasons") or [] if f == "length")
            t["usage_missing"] += int(bool(rec.get("usage_missing")))
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                t[k] += int(rec.get(k) or 0)
            t["latency_s"] = round(t["latency_s"] + float(rec.get("latency_s") or 0.0), 3)
            t["last_ts"] = rec["ts"]
            with open(self.req_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            self._write_totals_locked()

    def _write_totals_locked(self) -> None:
        tmp = self.tot_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.totals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.tot_path)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.totals))

    def flush(self) -> None:
        with self.lock:
            self._write_totals_locked()


def _client_label(auth: str | None) -> str:
    if auth and auth.lower().startswith("bearer "):
        label = auth[7:].strip()
        return label or "anonymous"
    return "anonymous"


def _usage_from_body(body: bytes) -> tuple[dict | None, list[str], str | None]:
    try:
        doc = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, [], None
    if not isinstance(doc, dict):
        return None, [], None
    finish = [c.get("finish_reason") for c in doc.get("choices") or [] if isinstance(c, dict)]
    return doc.get("usage") if isinstance(doc.get("usage"), dict) else None, finish, doc.get("model")


def make_handler(upstream: str, upstream_key: str, ledger: Ledger, timeout: float):
    up = urlsplit(upstream)
    up_host, up_port = up.hostname or "127.0.0.1", up.port or 80

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ro27-token-proxy/1"

        def log_message(self, fmt, *args):  # quiet; the ledger is the log
            pass

        def _send_json(self, code: int, doc: dict) -> None:
            data = json.dumps(doc).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/proxy/totals"):
                return self._send_json(200, ledger.snapshot())
            return self._forward(b"")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self._forward(self.rfile.read(n) if n else b"")

        def _forward(self, body: bytes) -> None:
            t0 = time.time()
            client = _client_label(self.headers.get("Authorization"))
            req_doc = None
            if body:
                try:
                    req_doc = json.loads(body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    req_doc = None
            stream = bool(isinstance(req_doc, dict) and req_doc.get("stream"))
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP and k.lower() != "authorization"}
            if upstream_key:
                headers["Authorization"] = f"Bearer {upstream_key}"
            if body:
                headers["Content-Length"] = str(len(body))
            rec = {"ts": round(t0, 3), "client": client, "method": self.command, "path": self.path,
                   "stream": stream, "model": req_doc.get("model") if isinstance(req_doc, dict) else None}
            accounted = self.path.startswith("/v1/") and self.command == "POST"
            try:
                conn = http.client.HTTPConnection(up_host, up_port, timeout=timeout)
                conn.request(self.command, self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
            except OSError as exc:
                rec.update(status=502, error=f"{type(exc).__name__}: {exc}", latency_s=round(time.time() - t0, 3))
                if accounted:
                    ledger.record(rec)
                return self._send_json(502, {"error": {"message": f"token proxy: upstream unreachable: {exc}"}})
            def account(status, usage, finish):
                # Written BEFORE the client sees the end of the response, so a client that
                # reads totals right after its call always finds the call counted.
                if not accounted:
                    return
                rec.update(status=status, latency_s=round(time.time() - t0, 3), finish_reasons=finish)
                if usage:
                    rec.update(prompt_tokens=int(usage.get("prompt_tokens") or 0),
                               completion_tokens=int(usage.get("completion_tokens") or 0),
                               total_tokens=int(usage.get("total_tokens") or 0))
                elif 200 <= status < 300:
                    rec["usage_missing"] = True
                ledger.record(rec)

            try:
                if stream and resp.status == 200:
                    self._relay_stream(resp, account)
                else:
                    data = resp.read()
                    usage, finish, model = _usage_from_body(data)
                    if model:
                        rec["model"] = model
                    account(resp.status, usage, finish)
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in HOP_BY_HOP:
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            finally:
                conn.close()

        def _relay_stream(self, resp, account):
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            usage, finish, buf = None, [], b""
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    try:
                        ev = json.loads(line[5:].strip().decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(ev.get("usage"), dict):
                        usage = ev["usage"]
                    finish += [c.get("finish_reason") for c in ev.get("choices") or []
                               if isinstance(c, dict) and c.get("finish_reason")]
            account(resp.status, usage, finish)
            self.wfile.write(b"0\r\n\r\n")

    return Handler


def log_to_wandb(ledger: Ledger, args) -> None:
    key = os.environ.get("WANDB_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        print("token_proxy: no WANDB_API_KEY, W&B upload skipped", flush=True)
        return
    try:
        import wandb
    except ImportError:
        print("token_proxy: wandb not installed, W&B upload skipped", flush=True)
        return
    try:
        totals = ledger.snapshot()
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
                         job_type="token_proxy", name=f"{args.session_id}__proxy",
                         tags=[t for t in (args.session_id, "v2", "token_proxy") if t],
                         config={"session_id": args.session_id, "upstream": args.upstream,
                                 "listen_port": args.listen_port}, reinit=True)
        cols = ["client", "calls", "ok_calls", "error_calls", "http_400", "finish_length", "usage_missing",
                "prompt_tokens", "completion_tokens", "total_tokens", "latency_s"]
        table = wandb.Table(columns=cols)
        for client, t in sorted(totals.items()):
            table.add_data(client, *[t.get(c) for c in cols[1:]])
            for k in ("prompt_tokens", "completion_tokens", "calls"):
                run.summary[f"tokens/{client}/{k.replace('_tokens', '')}"] = t.get(k)
        run.log({"per_client_totals": table})
        art = wandb.Artifact(f"{args.session_id}-proxy-log", type="proxy-log")
        for p in (ledger.req_path, ledger.tot_path):
            if p.exists():
                art.add_file(str(p))
        run.log_artifact(art)
        run.finish()
        print(f"token_proxy: W&B run {args.session_id}__proxy logged ({len(totals)} clients)", flush=True)
    except Exception as exc:  # noqa: BLE001 - W&B must never take the proxy down
        print(f"token_proxy: W&B upload failed: {type(exc).__name__}: {exc}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=8090)
    ap.add_argument("--upstream", required=True, help="vLLM base URL, e.g. http://127.0.0.1:8020")
    ap.add_argument("--upstream-key-env", default="UPSTREAM_API_KEY",
                    help="env var holding the server key (never passed on the command line)")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--session-id", default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="PlanBench")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT_NAME") or "RogueOne-AGENT2027")
    ap.add_argument("--wandb-group", default="A08")
    args = ap.parse_args(argv)

    ledger = Ledger(Path(args.log_dir))
    handler = make_handler(args.upstream, os.environ.get(args.upstream_key_env, ""), ledger, args.timeout)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
    server.daemon_threads = True

    def _stop(signum, _frame):
        print(f"token_proxy: signal {signum}, shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(f"token_proxy: {args.listen_host}:{args.listen_port} -> {args.upstream} log={args.log_dir}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        ledger.flush()
        tot = ledger.snapshot()
        print("token_proxy: totals " + json.dumps(
            {c: [t["calls"], t["prompt_tokens"], t["completion_tokens"]] for c, t in tot.items()}), flush=True)
        if args.wandb:
            log_to_wandb(ledger, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
