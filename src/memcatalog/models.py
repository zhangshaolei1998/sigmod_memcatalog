"""Core data model for sessions, facts, plans, evidence, and experience."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class PlanClass(str, Enum):
    RETRIEVE = "retrieve"
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"


@dataclass(frozen=True)
class Turn:
    speaker: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(speaker=str(data["speaker"]), content=str(data["content"]))


@dataclass
class Session:
    session_id: str
    user_id: str
    timestamp: str
    turns: List[Turn]
    closed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(f"{turn.speaker}: {turn.content}" for turn in self.turns)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["turns"] = [turn.to_dict() for turn in self.turns]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=str(data["session_id"]),
            user_id=str(data["user_id"]),
            timestamp=str(data["timestamp"]),
            turns=[Turn.from_dict(turn) for turn in data.get("turns", [])],
            closed=bool(data.get("closed", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SourceSpan:
    session_id: str
    start: int
    end: int
    quote: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceSpan":
        return cls(
            session_id=str(data["session_id"]),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            quote=str(data.get("quote", "")),
        )


@dataclass
class FactRecord:
    fact_id: str
    user_id: str
    subject: str
    predicate: str
    object: str
    source_spans: List[SourceSpan]
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    event_id: Optional[str] = None
    supersedes: Optional[str] = None
    verified: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple:
        return self.subject.casefold().strip(), self.predicate.casefold().strip()

    def is_valid_at(self, timestamp: Optional[str]) -> bool:
        if timestamp is None:
            return self.valid_to is None
        if self.valid_from and self.valid_from > timestamp:
            return False
        return not self.valid_to or timestamp < self.valid_to

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_spans"] = [span.to_dict() for span in self.source_spans]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactRecord":
        return cls(
            fact_id=str(data["fact_id"]),
            user_id=str(data["user_id"]),
            subject=str(data["subject"]),
            predicate=str(data["predicate"]),
            object=str(data["object"]),
            source_spans=[SourceSpan.from_dict(span) for span in data.get("source_spans", [])],
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            event_id=data.get("event_id"),
            supersedes=data.get("supersedes"),
            verified=bool(data.get("verified", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Request:
    request_id: str
    kind: str
    target: str
    predicate: Optional[str] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Request":
        return cls(**data)


@dataclass(frozen=True)
class Binding:
    source_request: str
    target_request: str
    target_slot: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Binding":
        return cls(**data)


@dataclass
class QueryPlan:
    plan_id: str
    plan_class: PlanClass
    requests: List[Request]
    bindings: List[Binding]
    result_operator: str
    snapshot_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        request_ids = {request.request_id for request in self.requests}
        if len(request_ids) != len(self.requests):
            raise ValueError("request IDs must be unique")
        for binding in self.bindings:
            if binding.source_request not in request_ids or binding.target_request not in request_ids:
                raise ValueError("bindings must reference requests in the plan")
            if binding.source_request == binding.target_request:
                raise ValueError("self-bindings are not allowed")
            target = next(
                request for request in self.requests if request.request_id == binding.target_request
            )
            if binding.target_slot not in target.slots:
                raise ValueError("binding target slot is not declared by the request")
        indegree = {request_id: 0 for request_id in request_ids}
        outgoing = {request_id: [] for request_id in request_ids}
        for binding in self.bindings:
            outgoing[binding.source_request].append(binding.target_request)
            indegree[binding.target_request] += 1
        ready = [request_id for request_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            request_id = ready.pop()
            visited += 1
            for target_id in outgoing[request_id]:
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    ready.append(target_id)
        if visited != len(request_ids):
            raise ValueError("query-plan bindings must form an acyclic graph")
        if self.plan_class is PlanClass.RETRIEVE and (
            len(self.requests) != 1 or self.bindings
        ):
            raise ValueError("retrieve plans must contain one unbound request")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_class": self.plan_class.value,
            "requests": [request.to_dict() for request in self.requests],
            "bindings": [binding.to_dict() for binding in self.bindings],
            "result_operator": self.result_operator,
            "snapshot_at": self.snapshot_at,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceBundle:
    fact_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    derived: Dict[str, Any] = field(default_factory=dict)
    gaps: List[str] = field(default_factory=list)
    status: SupportStatus = SupportStatus.UNKNOWN

    def merge(self, other: "EvidenceBundle") -> "EvidenceBundle":
        return EvidenceBundle(
            fact_ids=list(dict.fromkeys(self.fact_ids + other.fact_ids)),
            session_ids=list(dict.fromkeys(self.session_ids + other.session_ids)),
            derived={**self.derived, **other.derived},
            gaps=list(dict.fromkeys(self.gaps + other.gaps)),
            status=_merge_status(self.status, other.status),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def _merge_status(left: SupportStatus, right: SupportStatus) -> SupportStatus:
    if left is SupportStatus.SUPPORTED and right is SupportStatus.SUPPORTED:
        return SupportStatus.SUPPORTED
    if left is SupportStatus.UNKNOWN and right is SupportStatus.UNKNOWN:
        return SupportStatus.UNKNOWN
    return SupportStatus.PARTIAL


@dataclass
class ActionRecord:
    kind: str
    context: str
    predicted_reduction: float
    observed_reduction: float
    call_cost: int
    reservation: int
    completed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionRecord:
    query_id: str
    user_id: str
    question: str
    context_key: str
    plan: Dict[str, Any]
    returned_fact_ids: List[str]
    returned_session_ids: List[str]
    status: str
    prompt: str
    answer: Optional[str]
    actions: List[ActionRecord]
    memory_calls: int
    complete_feedback: bool = False
    used_fact_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["actions"] = [action.to_dict() for action in self.actions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        return cls(
            query_id=str(data["query_id"]),
            user_id=str(data["user_id"]),
            question=str(data["question"]),
            context_key=str(data["context_key"]),
            plan=dict(data.get("plan", {})),
            returned_fact_ids=list(data.get("returned_fact_ids", [])),
            returned_session_ids=list(data.get("returned_session_ids", [])),
            status=str(data.get("status", SupportStatus.UNKNOWN.value)),
            prompt=str(data.get("prompt", "")),
            answer=data.get("answer"),
            actions=[ActionRecord(**action) for action in data.get("actions", [])],
            memory_calls=int(data.get("memory_calls", 0)),
            complete_feedback=bool(data.get("complete_feedback", False)),
            used_fact_ids=list(data.get("used_fact_ids", [])),
        )
