"""HTTP 接口层测试：在随机端口启动真实服务进行端到端验证。"""

import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import app.store as store_module
from app.models import parse_request
from app.scheduler import canonical_hash
from app.server import make_handler
from app.store import DecisionStore


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HttpServerFixture:
    def __init__(self) -> None:
        self.store = DecisionStore()
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", free_port()), make_handler(self.store)
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "HttpServerFixture":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def call(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())


BASE_PAYLOAD = {
    "audit_id": "API-001",
    "gate_period": 1000,
    "flows": [
        {"flow_id": "A", "priority": 0, "period": 1000,
         "transmit_time": 100, "deadline": 1000}
    ],
    "gate_entries": [{"start": 0, "end": 500, "priorities": [0]}],
}


class ApiTests(unittest.TestCase):
    def test_health_and_page(self):
        with HttpServerFixture() as srv:
            status, body = srv.call("GET", "/api/health")
            self.assertEqual(status, 200)
            self.assertTrue(body["ok"])
            req = urllib.request.Request(srv.base + "/")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("冻结", resp.read().decode())

    def test_submit_freeze_read_idempotent_conflict(self):
        with HttpServerFixture() as srv:
            status, body = srv.call("POST", "/api/submit", BASE_PAYLOAD)
            self.assertEqual(status, 201)
            self.assertTrue(body["created"])
            h1 = body["content_hash"]
            self.assertEqual(body["decision"]["verdict"], "SCHEDULABLE")

            # 完全相同的提交 -> 同一冻结结论。
            status, body = srv.call("POST", "/api/submit", BASE_PAYLOAD)
            self.assertEqual(status, 200)
            self.assertFalse(body["created"])
            self.assertEqual(body["content_hash"], h1)

            # 读取接口返回同一冻结结论。
            status, body = srv.call("GET", "/api/decisions/API-001")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], h1)
            self.assertTrue(body["frozen"])

            # 同一标识不同内容 -> 409，原裁决不变。
            changed = json.loads(json.dumps(BASE_PAYLOAD))
            changed["gate_entries"][0]["end"] = 400
            status, body = srv.call("POST", "/api/submit", changed)
            self.assertEqual(status, 409)
            self.assertEqual(body["error"], "AUDIT_CONFLICT")
            self.assertNotEqual(body["stored_hash"], body["incoming_hash"])
            self.assertEqual(body["existing_decision"]["content_hash"], h1)

            status, body = srv.call("GET", "/api/decisions/API-001")
            self.assertEqual(body["content_hash"], h1)
            self.assertEqual(
                body["request"]["gate_entries"][0]["end"], 500
            )

    def test_validation_error_and_missing(self):
        with HttpServerFixture() as srv:
            bad = json.loads(json.dumps(BASE_PAYLOAD))
            bad["audit_id"] = "API-BAD"
            bad["flows"] = []
            status, body = srv.call("POST", "/api/submit", bad)
            self.assertEqual(status, 400)
            self.assertEqual(body["error"], "VALIDATION_FAILED")

            status, body = srv.call("GET", "/api/decisions/UNKNOWN")
            self.assertEqual(status, 404)

            # 非 JSON 请求体 -> 400。
            req = urllib.request.Request(
                srv.base + "/api/submit",
                data=b"not-json",
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            try:
                urllib.request.urlopen(req, timeout=5)
                self.fail("应返回 400")
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 400)

    def test_non_convergent_payload_via_api(self):
        with HttpServerFixture() as srv:
            payload = {
                "audit_id": "API-GROW",
                "gate_period": 1000,
                "flows": [
                    {"flow_id": "X", "priority": 0, "period": 1000,
                     "transmit_time": 900, "deadline": 1000}
                ],
                "gate_entries": [{"start": 0, "end": 100, "priorities": [0]}],
            }
            status, body = srv.call("POST", "/api/submit", payload)
            self.assertEqual(status, 201)
            self.assertEqual(body["decision"]["verdict"], "NON_CONVERGENT")
            chain = body["decision"]["growth_chain"]
            self.assertGreaterEqual(len(chain), 2)


class GatedAdjudicate:
    """包装真实裁决：目标标识进入后阻塞，直到测试放行；记录全部进入次数。"""

    def __init__(self, block_ids: set[str] | None = None) -> None:
        self._real = store_module.adjudicate
        self._block_ids = block_ids
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, req):
        with self._lock:
            self.calls.append(req.audit_id)
        if self._block_ids is None or req.audit_id in self._block_ids:
            self.entered.set()
            self.release.wait(timeout=10)
        return self._real(req)


class FlakyAdjudicate:
    """每个目标标识的首次计算在放行后抛错，其余调用委托真实裁决。"""

    def __init__(self, fail_ids: set[str]) -> None:
        self._real = store_module.adjudicate
        self._fail_ids = set(fail_ids)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, req):
        with self._lock:
            self.calls.append(req.audit_id)
            should_fail = req.audit_id in self._fail_ids
            self._fail_ids.discard(req.audit_id)
        if should_fail:
            self.entered.set()
            self.release.wait(timeout=10)
            raise RuntimeError("模拟裁决计算故障")
        return self._real(req)


class ConcurrentSubmitTests(unittest.TestCase):
    """并发提交验收：同一审计标识任一时刻只能形成一份冻结结论。"""

    @staticmethod
    def _payload(audit_id: str, transmit_time: int = 100) -> dict:
        payload = json.loads(json.dumps(BASE_PAYLOAD))
        payload["audit_id"] = audit_id
        payload["flows"][0]["transmit_time"] = transmit_time
        return payload

    @staticmethod
    def _post(srv: HttpServerFixture, results: list, payload: dict) -> None:
        results.append(srv.call("POST", "/api/submit", payload))

    def test_concurrent_identical_freezes_once(self):
        # 两次等价请求在裁决尚未返回时交叠：仅一次创建，
        # 哈希、冻结时间与冻结结论完全一致。
        payload = self._payload("API-CONC-SAME")
        gate = GatedAdjudicate()
        with HttpServerFixture() as srv, mock.patch.object(
            store_module, "adjudicate", gate
        ):
            results: list = []
            t1 = threading.Thread(target=self._post, args=(srv, results, payload))
            t1.start()
            # 第一份请求已进入裁决计算且响应尚未返回。
            self.assertTrue(gate.entered.wait(timeout=5))
            t2 = threading.Thread(target=self._post, args=(srv, results, payload))
            t2.start()
            # 给第二份请求足够时间抵达并进入等待；此时任何响应都不得返回。
            time.sleep(0.3)
            self.assertEqual(results, [])
            # 等价重叠提交不得触发第二次裁决计算。
            self.assertEqual(gate.calls, ["API-CONC-SAME"])
            gate.release.set()
            t1.join(timeout=5)
            t2.join(timeout=5)

            self.assertEqual(len(results), 2)
            created = [body for status, body in results if status == 201]
            replayed = [body for status, body in results if status == 200]
            self.assertEqual(len(created), 1, f"应仅一次创建: {results}")
            self.assertEqual(len(replayed), 1, f"其余请求应回放同一结论: {results}")
            self.assertEqual(gate.calls, ["API-CONC-SAME"])
            self.assertTrue(created[0]["created"])
            self.assertFalse(replayed[0]["created"])
            for key in ("content_hash", "frozen_at", "request", "decision"):
                self.assertEqual(created[0][key], replayed[0][key], key)

            status, body = srv.call("GET", "/api/decisions/API-CONC-SAME")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], created[0]["content_hash"])
            self.assertEqual(body["frozen_at"], created[0]["frozen_at"])
            self.assertEqual(body["decision"], created[0]["decision"])

    def test_concurrent_conflict_echoes_frozen_original(self):
        # 第二份并发提交只改动一项业务参数：必须等在途裁决冻结后收到 409，
        # 且冲突响应回显已冻结的原结论。
        payload_x = self._payload("API-CONC-CONFLICT", transmit_time=100)
        payload_y = self._payload("API-CONC-CONFLICT", transmit_time=120)
        gate = GatedAdjudicate()
        with HttpServerFixture() as srv, mock.patch.object(
            store_module, "adjudicate", gate
        ):
            results: list = []
            t1 = threading.Thread(target=self._post, args=(srv, results, payload_x))
            t1.start()
            self.assertTrue(gate.entered.wait(timeout=5))
            t2 = threading.Thread(target=self._post, args=(srv, results, payload_y))
            t2.start()
            # 首份裁决可读取前，冲突方不得提前拿到缺少原裁决详情的响应。
            time.sleep(0.3)
            self.assertEqual(results, [])
            gate.release.set()
            t1.join(timeout=5)
            t2.join(timeout=5)

            self.assertEqual(len(results), 2)
            by_status = {status: body for status, body in results}
            self.assertIn(201, by_status)
            self.assertIn(409, by_status)
            created_body = by_status[201]
            conflict_body = by_status[409]
            self.assertEqual(conflict_body["error"], "AUDIT_CONFLICT")
            self.assertEqual(
                conflict_body["stored_hash"],
                canonical_hash(parse_request(payload_x)),
            )
            self.assertEqual(
                conflict_body["incoming_hash"],
                canonical_hash(parse_request(payload_y)),
            )
            existing = conflict_body["existing_decision"]
            self.assertIsNotNone(existing, "冲突响应必须回显已冻结的原结论")
            self.assertEqual(existing["content_hash"], created_body["content_hash"])
            self.assertEqual(existing["frozen_at"], created_body["frozen_at"])
            self.assertEqual(existing["decision"], created_body["decision"])
            self.assertEqual(existing["request"]["flows"][0]["transmit_time"], 100)

            status, body = srv.call("GET", "/api/decisions/API-CONC-CONFLICT")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], created_body["content_hash"])
            self.assertEqual(body["request"]["flows"][0]["transmit_time"], 100)

    def test_concurrent_failure_releases_waiter_and_id(self):
        # 计算途中失败：等待中的等价请求被唤醒后重试并成功冻结，
        # 失败方收到 500，标识不被永久占用。
        payload = self._payload("API-CONC-FAIL")
        flaky = FlakyAdjudicate({"API-CONC-FAIL"})
        with HttpServerFixture() as srv, mock.patch.object(
            store_module, "adjudicate", flaky
        ):
            results: list = []
            t1 = threading.Thread(target=self._post, args=(srv, results, payload))
            t1.start()
            self.assertTrue(flaky.entered.wait(timeout=5))
            t2 = threading.Thread(target=self._post, args=(srv, results, payload))
            t2.start()
            time.sleep(0.3)
            self.assertEqual(results, [])
            flaky.release.set()
            t1.join(timeout=5)
            t2.join(timeout=5)

            self.assertEqual(len(results), 2)
            by_status = {status: body for status, body in results}
            self.assertIn(500, by_status)
            self.assertEqual(by_status[500]["error"], "ADJUDICATION_FAILED")
            self.assertIn(201, by_status)
            self.assertTrue(by_status[201]["created"])
            # 首次失败一次 + 等待者重试一次，共两次计算。
            self.assertEqual(flaky.calls, ["API-CONC-FAIL", "API-CONC-FAIL"])

            status, body = srv.call("GET", "/api/decisions/API-CONC-FAIL")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], by_status[201]["content_hash"])

    def test_sequential_failure_then_retry(self):
        # 顺序场景：计算失败返回 500 后，同内容重试可正常冻结；
        # 另一标识失败后改交不同内容同样可冻结（标识未被失败占用）。
        flaky = FlakyAdjudicate({"API-SEQ-FAIL-1", "API-SEQ-FAIL-2"})
        flaky.release.set()  # 不阻塞，首次计算立即失败
        with HttpServerFixture() as srv, mock.patch.object(
            store_module, "adjudicate", flaky
        ):
            payload = self._payload("API-SEQ-FAIL-1")
            status, body = srv.call("POST", "/api/submit", payload)
            self.assertEqual(status, 500)
            self.assertEqual(body["error"], "ADJUDICATION_FAILED")
            status, body = srv.call("GET", "/api/decisions/API-SEQ-FAIL-1")
            self.assertEqual(status, 404)

            status, body = srv.call("POST", "/api/submit", payload)
            self.assertEqual(status, 201)
            self.assertTrue(body["created"])

            other = self._payload("API-SEQ-FAIL-2", transmit_time=100)
            status, body = srv.call("POST", "/api/submit", other)
            self.assertEqual(status, 500)
            changed = self._payload("API-SEQ-FAIL-2", transmit_time=120)
            status, body = srv.call("POST", "/api/submit", changed)
            self.assertEqual(status, 201)
            self.assertEqual(
                body["content_hash"], canonical_hash(parse_request(changed))
            )

    def test_sequential_idempotent_submit_and_read(self):
        payload = self._payload("API-SEQ-IDEM")
        with HttpServerFixture() as srv:
            status, first = srv.call("POST", "/api/submit", payload)
            self.assertEqual(status, 201)
            self.assertTrue(first["created"])
            status, second = srv.call("POST", "/api/submit", payload)
            self.assertEqual(status, 200)
            self.assertFalse(second["created"])
            for key in ("content_hash", "frozen_at", "request", "decision"):
                self.assertEqual(first[key], second[key], key)
            status, body = srv.call("GET", "/api/decisions/API-SEQ-IDEM")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], first["content_hash"])
            self.assertEqual(body["frozen_at"], first["frozen_at"])

    def test_different_audit_ids_are_independent(self):
        # 一个标识的裁决计算进行中，不阻塞其他标识的提交与冻结。
        payload_a = self._payload("API-CONC-A")
        payload_b = self._payload("API-CONC-B")
        gate = GatedAdjudicate(block_ids={"API-CONC-A"})
        with HttpServerFixture() as srv, mock.patch.object(
            store_module, "adjudicate", gate
        ):
            results: list = []
            t1 = threading.Thread(target=self._post, args=(srv, results, payload_a))
            t1.start()
            self.assertTrue(gate.entered.wait(timeout=5))
            status, body_b = srv.call("POST", "/api/submit", payload_b)
            self.assertEqual(status, 201)
            self.assertTrue(body_b["created"])
            gate.release.set()
            t1.join(timeout=5)

            self.assertEqual(len(results), 1)
            status_a, body_a = results[0]
            self.assertEqual(status_a, 201)
            self.assertNotEqual(body_a["content_hash"], body_b["content_hash"])


if __name__ == "__main__":
    unittest.main()
