"""并发冻结语义验收：交叠提交、异内容冲突、失败重试与顺序幂等。

通过可控裁决钩子（进入时通知、放行前阻塞）确定性地安排两次请求在裁决
尚未返回时交叠，验证同一审计标识在任一时刻只形成一份冻结结论：
- 等价交叠提交仅一次创建，其余回放同一份结论（哈希/冻结时间完全一致）；
- 异内容并发提交在原结论冻结后收到 409，且响应回显完整原裁决；
- 计算途中失败不占用标识，等待者接管、后续重试正常完成；
- 不同审计标识并行处理，互不阻塞。
"""

import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app.models import parse_request
from app.scheduler import adjudicate
from app.server import make_handler
from app.store import ConflictError, DecisionStore


def make_payload(audit_id: str = "CONC-001", transmit_time: int = 100) -> dict:
    return {
        "audit_id": audit_id,
        "gate_period": 1000,
        "flows": [
            {"flow_id": "A", "priority": 0, "period": 1000,
             "transmit_time": transmit_time, "deadline": 1000}
        ],
        "gate_entries": [{"start": 0, "end": 500, "priorities": [0]}],
    }


class ControlledAdjudicator:
    """包装真实裁决：进入时计数并通知，放行前阻塞。

    - 用于确定性地让两次提交在“裁决尚未返回”时交叠；
    - fail_calls 指定的调用序号抛 RuntimeError，模拟计算途中失败。
    """

    def __init__(self, fail_calls: set[int] | None = None):
        self._lock = threading.Lock()
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self._fail_calls = set(fail_calls or ())

    def __call__(self, req):
        with self._lock:
            self.calls += 1
            n = self.calls
        self.entered.set()
        self.release.wait(timeout=10)
        if n in self._fail_calls:
            raise RuntimeError(f"注入的第 {n} 次裁决失败")
        return adjudicate(req)

    def wait_calls(self, n: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.calls >= n:
                    return True
            time.sleep(0.01)
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HttpServerFixture:
    def __init__(self, store: DecisionStore | None = None):
        self.store = store if store is not None else DecisionStore()
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
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())


class StoreOverlapTests(unittest.TestCase):
    """存储层：用可控裁决钩子制造确定性交叠。"""

    def test_overlapping_identical_single_creation(self):
        gate = ControlledAdjudicator()
        store = DecisionStore(adjudicator=gate)
        req = parse_request(make_payload())
        results, errors = [], []

        def worker():
            try:
                results.append(store.submit(req))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=worker)
        t1.start()
        self.assertTrue(gate.entered.wait(5), "领导者应已进入裁决计算")
        t2 = threading.Thread(target=worker)
        t2.start()
        time.sleep(0.3)  # 让跟随者充分进入等待，形成确定性交叠
        self.assertEqual(results, [], "裁决未返回前两个请求都不应完成")
        gate.release.set()
        t1.join(5)
        t2.join(5)
        self.assertFalse(t1.is_alive() or t2.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        created = sorted(c for _, c in results)
        self.assertEqual(created, [False, True], "等价交叠提交只能新建一次")
        d1, d2 = results[0][0], results[1][0]
        self.assertIs(d1, d2, "两份响应必须是同一份冻结结论")
        self.assertEqual(d1.content_hash, d2.content_hash)
        self.assertEqual(d1.frozen_at, d2.frozen_at)
        self.assertIs(store.get(req.audit_id), d1)
        self.assertEqual(store.list_ids(), [req.audit_id])

    def test_overlapping_different_content_conflict_after_freeze(self):
        gate = ControlledAdjudicator()
        store = DecisionStore(adjudicator=gate)
        req1 = parse_request(make_payload())
        req2 = parse_request(make_payload(transmit_time=120))
        outcome = {}

        def leader():
            outcome["leader"] = store.submit(req1)

        def follower():
            try:
                store.submit(req2)
                outcome["follower"] = None
            except ConflictError as exc:
                outcome["follower"] = exc

        t1 = threading.Thread(target=leader)
        t1.start()
        self.assertTrue(gate.entered.wait(5))
        t2 = threading.Thread(target=follower)
        t2.start()
        time.sleep(0.3)
        self.assertNotIn(
            "follower", outcome, "异内容跟随者必须等原结论冻结后才返回冲突"
        )
        gate.release.set()
        t1.join(5)
        t2.join(5)

        decision, created = outcome["leader"]
        self.assertTrue(created)
        exc = outcome["follower"]
        self.assertIsInstance(exc, ConflictError)
        self.assertEqual(exc.stored_hash, decision.content_hash)
        self.assertNotEqual(exc.incoming_hash, exc.stored_hash)
        # 冲突发生时原裁决已冻结可读。
        frozen = store.get(req1.audit_id)
        self.assertIsNotNone(frozen)
        self.assertEqual(frozen.content_hash, exc.stored_hash)

    def test_failed_adjudication_frees_id_for_retry(self):
        gate = ControlledAdjudicator(fail_calls={1})
        gate.release.set()
        store = DecisionStore(adjudicator=gate)
        req = parse_request(make_payload())
        with self.assertRaises(RuntimeError):
            store.submit(req)
        # 失败不占用标识：无冻结结论、无在途记录。
        self.assertIsNone(store.get(req.audit_id))
        self.assertEqual(store.list_ids(), [])
        # 合法重试正常完成并冻结。
        decision, created = store.submit(req)
        self.assertTrue(created)
        self.assertIs(store.get(req.audit_id), decision)

    def test_waiting_follower_takes_over_after_leader_failure(self):
        gate = ControlledAdjudicator(fail_calls={1})
        store = DecisionStore(adjudicator=gate)
        req = parse_request(make_payload())
        results, errors = [], []

        def worker():
            try:
                results.append(store.submit(req))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=worker)
        t1.start()
        self.assertTrue(gate.entered.wait(5))
        t2 = threading.Thread(target=worker)
        t2.start()
        time.sleep(0.3)
        gate.release.set()
        t1.join(5)
        t2.join(5)

        # 领导者计算失败，等待中的等价跟随者接管并成功冻结。
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(results), 1)
        decision, created = results[0]
        self.assertTrue(created)
        self.assertIs(store.get(req.audit_id), decision)

    def test_different_ids_processed_independently(self):
        gate = ControlledAdjudicator()
        store = DecisionStore(adjudicator=gate)
        req1 = parse_request(make_payload("CONC-A"))
        req2 = parse_request(make_payload("CONC-B"))
        results = {}

        def worker(key, req):
            results[key] = store.submit(req)

        t1 = threading.Thread(target=worker, args=("a", req1))
        t2 = threading.Thread(target=worker, args=("b", req2))
        t1.start()
        t2.start()
        # 两个标识的计算都进入在途状态（并行），而非被全局锁串行化。
        self.assertTrue(gate.wait_calls(2), "不同审计标识应能并行计算")
        gate.release.set()
        t1.join(5)
        t2.join(5)

        self.assertTrue(results["a"][1])
        self.assertTrue(results["b"][1])
        self.assertNotEqual(
            results["a"][0].content_hash, results["b"][0].content_hash
        )
        self.assertEqual(store.list_ids(), ["CONC-A", "CONC-B"])

    def test_sequential_idempotent_submit_and_read(self):
        store = DecisionStore()
        req = parse_request(make_payload("CONC-SEQ"))
        d1, c1 = store.submit(req)
        d2, c2 = store.submit(req)
        self.assertTrue(c1)
        self.assertFalse(c2)
        self.assertIs(d1, d2)
        self.assertIs(store.get("CONC-SEQ"), d1)
        # 顺序异内容 -> 冲突且原裁决不变。
        with self.assertRaises(ConflictError):
            store.submit(parse_request(make_payload("CONC-SEQ", transmit_time=120)))
        self.assertIs(store.get("CONC-SEQ"), d1)


class HttpConcurrencyTests(unittest.TestCase):
    """HTTP 层：真实接口下的交叠、冲突与失败重试。"""

    def test_http_overlap_identical_one_creation(self):
        gate = ControlledAdjudicator()
        store = DecisionStore(adjudicator=gate)
        with HttpServerFixture(store) as srv:
            responses = []

            def poster():
                responses.append(srv.call("POST", "/api/submit", make_payload()))

            t1 = threading.Thread(target=poster)
            t1.start()
            self.assertTrue(gate.entered.wait(5))
            t2 = threading.Thread(target=poster)
            t2.start()
            time.sleep(0.3)
            self.assertEqual(responses, [], "裁决未返回前两个请求都应挂起")
            gate.release.set()
            t1.join(5)
            t2.join(5)

            self.assertEqual(len(responses), 2)
            statuses = sorted(r[0] for r in responses)
            self.assertEqual(statuses, [200, 201], "等价交叠提交应恰有一次 201 新建")
            bodies = [r[1] for r in responses]
            self.assertEqual(sorted(b["created"] for b in bodies), [False, True])
            hashes = {b["content_hash"] for b in bodies}
            self.assertEqual(len(hashes), 1, "哈希必须完全一致")
            frozen_ats = {b["frozen_at"] for b in bodies}
            self.assertEqual(len(frozen_ats), 1, "冻结时间必须完全一致（同一份结论）")
            decisions = {json.dumps(b["decision"], sort_keys=True) for b in bodies}
            self.assertEqual(len(decisions), 1, "冻结结论必须完全一致")

            status, body = srv.call("GET", "/api/decisions/CONC-001")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], hashes.pop())
            self.assertTrue(body["frozen"])

    def test_http_overlap_conflict_echoes_frozen_decision(self):
        gate = ControlledAdjudicator()
        store = DecisionStore(adjudicator=gate)
        with HttpServerFixture(store) as srv:
            outcome = {}

            def leader():
                outcome["leader"] = srv.call("POST", "/api/submit", make_payload())

            def follower():
                outcome["follower"] = srv.call(
                    "POST", "/api/submit", make_payload(transmit_time=120)
                )

            t1 = threading.Thread(target=leader)
            t1.start()
            self.assertTrue(gate.entered.wait(5))
            t2 = threading.Thread(target=follower)
            t2.start()
            time.sleep(0.3)
            self.assertNotIn(
                "follower", outcome,
                "异内容请求不得在首份裁决可读取前提前收到冲突",
            )
            gate.release.set()
            t1.join(5)
            t2.join(5)

            status, body = outcome["leader"]
            self.assertEqual(status, 201)
            frozen_hash = body["content_hash"]

            status, body = outcome["follower"]
            self.assertEqual(status, 409)
            self.assertEqual(body["error"], "AUDIT_CONFLICT")
            self.assertEqual(body["stored_hash"], frozen_hash)
            self.assertNotEqual(body["incoming_hash"], frozen_hash)
            existing = body.get("existing_decision")
            self.assertIsNotNone(existing, "冲突响应必须回显已冻结的原裁决")
            self.assertEqual(existing["content_hash"], frozen_hash)
            self.assertEqual(existing["decision"]["verdict"], "SCHEDULABLE")

            # 原裁决保持不变。
            status, body = srv.call("GET", "/api/decisions/CONC-001")
            self.assertEqual(status, 200)
            self.assertEqual(body["content_hash"], frozen_hash)
            self.assertEqual(body["request"]["flows"][0]["transmit_time"], 100)

    def test_http_failure_then_retry(self):
        gate = ControlledAdjudicator(fail_calls={1})
        gate.release.set()
        store = DecisionStore(adjudicator=gate)
        with HttpServerFixture(store) as srv:
            status, body = srv.call("POST", "/api/submit", make_payload())
            self.assertEqual(status, 500)
            self.assertEqual(body["error"], "ADJUDICATION_FAILED")

            # 失败不占用标识：合法重试正常冻结。
            status, body = srv.call("POST", "/api/submit", make_payload())
            self.assertEqual(status, 201)
            self.assertTrue(body["created"])
            frozen_hash = body["content_hash"]

            status, read = srv.call("GET", "/api/decisions/CONC-001")
            self.assertEqual(status, 200)
            self.assertEqual(read["content_hash"], frozen_hash)

            # 冻结后等价提交保持幂等。
            status, again = srv.call("POST", "/api/submit", make_payload())
            self.assertEqual(status, 200)
            self.assertFalse(again["created"])
            self.assertEqual(again["content_hash"], frozen_hash)


if __name__ == "__main__":
    unittest.main()
