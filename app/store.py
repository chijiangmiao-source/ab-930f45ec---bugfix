"""冻结裁决存储。

语义：
- 审计标识首次提交：计算裁决并永久冻结（进程生命周期内不可变）。
- 同标识 + 内容完全相同（规范哈希一致）：返回同一份冻结结论（幂等）。
- 同标识 + 内容不同：409 冲突，原裁决保持不变。
- 读取接口永远只返回已冻结的结论。

并发语义（同一审计标识任一时刻至多一份冻结结论）：
- 等价的重叠提交由首个到达者计算，其余请求阻塞等待，结论冻结后回放
  同一份结果（仅一次创建，哈希与冻结时间完全一致）；
- 内容不同的重叠提交同样等待在途计算落定，随后以 409 冲突回显
  已冻结的原结论，绝不返回缺少原裁决详情的冲突响应；
- 计算途中失败会立即释放在途占位并唤醒等待者，该标识不被永久占用，
  后续合法重试可正常完成；
- 不同审计标识各有独立的在途占位，互不阻塞、独立处理。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

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
    """在途计算占位：首个到达者负责计算，其余提交在 done 上等待落定。"""

    audit_id: str
    content_hash: str
    started_at: float
    done: threading.Event = field(
        default_factory=threading.Event, compare=False
    )


class DecisionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decisions: dict[str, FrozenDecision] = {}
        self._inflight: dict[str, InFlightDecision] = {}

    def submit(self, req: ScheduleRequest) -> tuple[FrozenDecision, bool]:
        """提交并冻结。返回 (裁决, 是否本次新建)。冲突时抛 ConflictError。"""
        content_hash = canonical_hash(req)
        while True:
            with self._lock:
                existing = self._decisions.get(req.audit_id)
                if existing is not None:
                    if existing.content_hash != content_hash:
                        raise ConflictError(
                            req.audit_id, existing.content_hash, content_hash
                        )
                    return existing, False
                slot = self._inflight.get(req.audit_id)
                if slot is None:
                    # 首个到达者：占位于锁内，计算在锁外，独占创建权。
                    slot = InFlightDecision(req.audit_id, content_hash, time.time())
                    self._inflight[req.audit_id] = slot
                    break
            # 已有在途计算（无论内容是否相同）：等它落定后回到锁内重判——
            # 成功则等价者回放同一冻结结论、异内容者拿到已冻结的原结论；
            # 失败则占位已释放，重新竞争计算权。
            slot.done.wait()

        try:
            verdict = adjudicate(req)
        except BaseException:
            # 计算失败不得永久占用标识：释放占位并唤醒等待者重试。
            with self._lock:
                if self._inflight.get(req.audit_id) is slot:
                    self._inflight.pop(req.audit_id, None)
                slot.done.set()
            raise

        decision = FrozenDecision(
            audit_id=req.audit_id,
            content_hash=content_hash,
            frozen_at=time.time(),
            request=req.to_json(),
            verdict=verdict,
        )
        with self._lock:
            self._inflight.pop(req.audit_id, None)
            self._decisions[req.audit_id] = decision
            slot.done.set()
        return decision, True

    def get(self, audit_id: str) -> FrozenDecision | None:
        with self._lock:
            return self._decisions.get(audit_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._decisions)
