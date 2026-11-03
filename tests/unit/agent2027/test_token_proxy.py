"""X5 token proxy: pass-through, key swap, per-client token totals equal the upstream's own log."""

from __future__ import annotations

import http.client
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("token_proxy", REPO / "scripts" / "agent2027" / "token_proxy" / "token_proxy.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

SERVER_KEY = "server-secret"


class FakeVLLM:
    """Stands in for vLLM: checks the bearer key, returns a usage block, logs what it served."""

    def __init__(self):
        self.log: list[dict] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, doc, ctype="application/json"):
                data = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    return self._send(200, b"", "text/plain")
                return self._send(404, {"error": "nope"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("Authorization") != f"Bearer {SERVER_KEY}":
                    return self._send(401, {"error": "bad key"})
                text = body["messages"][0]["content"]
                if text == "too long":
                    fake.log.append({"status": 400})
                    return self._send(400, {"error": {"message": "context length"}})
                p, c = len(text) + 7, (len(text) * 3) % 50 + 1
                fake.log.append({"status": 200, "prompt": p, "completion": c})
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    evs = [{"choices": [{"delta": {"content": "x"}, "finish_reason": None}]},
                           {"choices": [{"delta": {}, "finish_reason": "length"}]},
                           {"choices": [], "usage": {"prompt_tokens": p, "completion_tokens": c,
                                                     "total_tokens": p + c}}]
                    for ev in evs:
                        line = f"data: {json.dumps(ev)}\n\n".encode()
                        self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
                    return
                return self._send(200, {"model": body.get("model"), "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture()
def proxy(tmp_path):
    up = FakeVLLM()
    ledger = tp.Ledger(tmp_path / "proxy")
    handler = tp.make_handler(f"http://127.0.0.1:{up.port}", SERVER_KEY, ledger, 30)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield up, ledger, srv.server_address[1]
    srv.shutdown()
    up.server.shutdown()


def _call(port, key, content, stream=False, path="/v1/chat/completions"):
    # LLM-FE's own request shape (llmfe/sampler.py: one HTTPConnection per call).
    conn = http.client.HTTPConnection(f"127.0.0.1:{port}")  # LLM-FE passes "localhost:<port>"
    payload = {"max_tokens": 2048, "model": "openai/gpt-oss-120b",
               "messages": [{"role": "user", "content": content}]}
    if stream:
        payload["stream"] = True
    conn.request("POST", path, json.dumps(payload),
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, data


def test_ten_calls_totals_match_upstream(proxy):
    up, ledger, port = proxy
    keys = ["a08-llmfe-synp_p2-r0", "a08-llmfe-synp_p3-r1"]
    for i in range(10):
        status, data = _call(port, keys[i % 2], "prompt number %d %s" % (i, "x" * i))
        assert status == 200 and json.loads(data)["choices"][0]["message"]["content"] == "ok"
    tot = json.loads((ledger.log_dir / "totals.json").read_text())
    served = [r for r in up.log if r["status"] == 200]
    assert len(served) == 10
    assert sum(t["prompt_tokens"] for t in tot.values()) == sum(r["prompt"] for r in served)
    assert sum(t["completion_tokens"] for t in tot.values()) == sum(r["completion"] for r in served)
    assert {k: tot[k]["calls"] for k in keys} == {keys[0]: 5, keys[1]: 5}
    # per client: the upstream log is in call order, clients alternate
    for j, k in enumerate(keys):
        assert tot[k]["prompt_tokens"] == sum(r["prompt"] for r in served[j::2])
    lines = (ledger.log_dir / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 10 and all(json.loads(x)["status"] == 200 for x in lines)
    # the server key never leaves the proxy
    assert SERVER_KEY not in (ledger.log_dir / "requests.jsonl").read_text()


def test_health_forwarded_and_not_counted(proxy):
    up, ledger, port = proxy
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/health")
    assert conn.getresponse().status == 200
    conn.close()
    assert ledger.snapshot() == {}


def test_errors_and_stream_usage(proxy):
    up, ledger, port = proxy
    status, _ = _call(port, "k1", "too long")
    assert status == 400
    status, data = _call(port, "k1", "abc", stream=True)
    assert status == 200 and b"[DONE]" in data
    t = ledger.snapshot()["k1"]
    assert t["calls"] == 2 and t["http_400"] == 1 and t["error_calls"] == 1
    assert t["finish_length"] == 1 and t["usage_missing"] == 0
    served = [r for r in up.log if r["status"] == 200]
    assert t["prompt_tokens"] == served[0]["prompt"] and t["completion_tokens"] == served[0]["completion"]


def test_live_totals_endpoint_and_unreachable_upstream(tmp_path):
    ledger = tp.Ledger(tmp_path / "p")
    handler = tp.make_handler("http://127.0.0.1:9", SERVER_KEY, ledger, 2)  # nothing listens on 9
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    status, _ = _call(port, "k2", "hi")
    assert status == 502
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/proxy/totals")
    doc = json.loads(conn.getresponse().read())
    conn.close()
    srv.shutdown()
    assert doc["k2"]["error_calls"] == 1 and doc["k2"]["prompt_tokens"] == 0
