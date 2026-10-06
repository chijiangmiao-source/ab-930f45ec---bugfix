"""冻结裁决存储。

语义：
- 同一审计标识在任一时刻只能形成一份冻结结论（进程生命周期内不可变）。
- 并发等价提交：由最先到达的请求负责计算（领导者），其余请求等待结论
  冻结后回放同一份内容（created=False），哈希、冻结时间与结论完全一致。
- 并发异内容提交：等待在途计算落地后明确冲突（409），此时原裁决已冻结
  可读，冲突响应可回显完整原裁决；原裁决保持不变。
- 计算途中失败：释放该标识并唤醒等待者，后续合法重试可正常完成。
- 不同审计标识互不阻塞，各自独立计算。
- 读取接口永远只返回已冻结的结论。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .models import ScheduleRequest
from .scheduler import adjudicate, canonical_hash


class ConflictError(Exception):
    def __init__(self, audit_id: str, stored_hash: str, incoming_hash: str):
        self.audit_id = audit_id
        self.stored_hash = stored_hash
        self.incoming_hash = incoming_hash
        super().__init__(f"审计标识 {audit_id} 已冻结不同内容的裁决")


@dataclass(frozen=True)
class FrozenDecision:
    audit_id: str
    content_hash: str
    frozen_at: float
    request: dict[str, Any]
    verdict: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "content_hash": self.content_hash,
            "frozen_at": self.frozen_at,
            "frozen": True,
            "request": self.request,
            "decision": self.verdict,
        }


@dataclass(frozen=True)
class InFlightDecision:
    audit_id: str
    content_hash: str
    started_at: float


class DecisionStore:
    def __init__(
        self,
        adjudicator: Callable[[ScheduleRequest], dict[str, Any]] | None = None,
    ) -> None:
        # 单一条件变量协调所有标识：锁只保护簿记，裁决计算在锁外进行，
        # 因此不同审计标识始终可以并行处理。
        self._cond = threading.Condition()
        self._decisions: dict[str, FrozenDecision] = {}
        self._inflight: dict[str, InFlightDecision] = {}
        self._adjudicator = adjudicator if adjudicator is not None else adjudicate

    def submit(self, req: ScheduleRequest) -> tuple[FrozenDecision, bool]:
        """提交并冻结。返回 (裁决, 是否本次新建)。冲突时抛 ConflictError。

        同一标识已有在途计算时，其余提交一律等待其落地：等价内容回放同一
        份冻结结论，异内容在原结论冻结后抛出冲突（原裁决随即可读）。
        """
        content_hash = canonical_hash(req)
        with self._cond:
            while True:
                existing = self._decisions.get(req.audit_id)
                if existing is not None:
                    if existing.content_hash != content_hash:
                        raise ConflictError(
                            req.audit_id, existing.content_hash, content_hash
                        )
                    return existing, False
                if req.audit_id in self._inflight:
                    # 已有在途计算：等其冻结或失败释放后再判定，保证任一时刻
                    # 只形成一份结论，且异内容冲突一定能拿到已冻结原裁决。
                    self._cond.wait()
                    continue
                # 成为领导者：登记在途计算，随后在锁外执行裁决。
                self._inflight[req.audit_id] = InFlightDecision(
                    req.audit_id, content_hash, time.time()
                )
                break

        try:
            verdict = self._adjudicator(req)
        except BaseException:
            # 计算失败不得永久占用标识：清理在途记录并唤醒等待者，
            # 由等待者接管或后续重试重新计算。
            with self._cond:
                self._inflight.pop(req.audit_id, None)
                self._cond.notify_all()
            raise

        decision = FrozenDecision(
            audit_id=req.audit_id,
            content_hash=content_hash,
            frozen_at=time.time(),
            request=req.to_json(),
            verdict=verdict,
        )
        with self._cond:
            self._inflight.pop(req.audit_id, None)
            self._decisions[req.audit_id] = decision
            self._cond.notify_all()
        return decision, True

    def get(self, audit_id: str) -> FrozenDecision | None:
        with self._cond:
            return self._decisions.get(audit_id)

    def list_ids(self) -> list[str]:
        with self._cond:
            return sorted(self._decisions)
