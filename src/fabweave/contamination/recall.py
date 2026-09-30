from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping

from fabweave.core.ids import deterministic_id


class LinkKind(str, Enum):
    SPLIT = "split"
    MERGE = "merge"


class MaterialState(str, Enum):
    WIP = "wip"
    SHIPPED = "shipped"
    DISPOSED = "disposed"


@dataclass(frozen=True, slots=True)
class ContaminationLink:
    kind: LinkKind
    sources: tuple[str, ...]
    target: str


@dataclass(frozen=True, slots=True)
class CarrierAssociation:
    carrier_id: str
    object_ids: tuple[str, ...]


def split_link(source: str, target: str) -> ContaminationLink:
    return ContaminationLink(LinkKind.SPLIT, (source,), target)


def merge_link(sources: list[str] | tuple[str, ...], target: str) -> ContaminationLink:
    return ContaminationLink(LinkKind.MERGE, tuple(sources), target)


def carrier_association(carrier_id: str, object_ids: list[str] | tuple[str, ...]) -> CarrierAssociation:
    return CarrierAssociation(carrier_id, tuple(object_ids))


@dataclass(frozen=True, slots=True)
class MaterialObject:
    object_id: str
    state: MaterialState = MaterialState.WIP


class ContaminationTraceGraph:
    """Split/merge genealogy plus undirected shared-carrier relationships."""

    def __init__(
        self,
        links: list[ContaminationLink] | tuple[ContaminationLink, ...] = (),
        carrier_associations: list[CarrierAssociation] | tuple[CarrierAssociation, ...] = (),
        object_ids: set[str] | tuple[str, ...] = (),
    ) -> None:
        material_ids: set[str] = set(object_ids)
        for link in links:
            material_ids.update(link.sources)
            material_ids.add(link.target)
        for association in carrier_associations:
            material_ids.update(association.object_ids)

        carrier_ids = {association.carrier_id for association in carrier_associations}
        overlap = sorted(material_ids & carrier_ids)
        if overlap:
            raise ValueError(f"carrier id conflicts with material id: {', '.join(overlap)}")

        self._materials = frozenset(material_ids)
        self._carriers = frozenset(carrier_ids)
        self._upstream: dict[str, set[str]] = defaultdict(set)
        self._downstream: dict[str, set[str]] = defaultdict(set)
        self._carrier_members: dict[str, set[str]] = defaultdict(set)
        self._carriers_by_object: dict[str, set[str]] = defaultdict(set)
        self._merge_peers: dict[str, set[str]] = defaultdict(set)

        for link in links:
            sources = tuple(dict.fromkeys(link.sources))
            if not sources or link.target in sources:
                raise ValueError("contamination link must connect distinct objects")
            if len(sources) != len(link.sources):
                raise ValueError("contamination link sources must be unique")
            endpoints = (*sources, link.target)
            missing = sorted(object_id for object_id in endpoints if object_id not in self._materials)
            if missing:
                raise ValueError(f"unknown contamination trace object: {', '.join(missing)}")
            if link.kind is LinkKind.SPLIT and len(sources) != 1:
                raise ValueError("a split link must have exactly one source")
            if link.kind is LinkKind.MERGE and len(sources) < 2:
                raise ValueError("a merge link must have at least two sources")
            for source in sources:
                self._upstream[link.target].add(source)
                self._downstream[source].add(link.target)
                if link.kind is LinkKind.MERGE:
                    self._merge_peers[source].update(member for member in sources if member != source)

        for association in carrier_associations:
            members = tuple(dict.fromkeys(association.object_ids))
            if not members:
                raise ValueError("a carrier association must contain at least one object")
            if len(members) != len(association.object_ids):
                raise ValueError("an object can only be listed once per carrier association")
            missing = sorted(object_id for object_id in members if object_id not in self._materials)
            if missing:
                raise ValueError(f"unknown contamination trace object: {', '.join(missing)}")
            self._carrier_members[association.carrier_id].update(members)
            for object_id in members:
                self._carriers_by_object[object_id].add(association.carrier_id)

    @property
    def object_ids(self) -> frozenset[str]:
        return self._materials

    @property
    def carrier_ids(self) -> frozenset[str]:
        return self._carriers

    def related(
        self,
        start: str,
        direction: Literal["upstream", "downstream", "both"],
        max_depth: int,
    ) -> tuple[str, ...]:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if direction not in {"upstream", "downstream", "both"}:
            raise ValueError("invalid traversal direction")
        if start not in self._materials:
            raise ValueError(f"unknown contamination trace object: {start}")

        visited = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth == max_depth:
                continue
            neighbors: set[str] = set()
            if direction in {"upstream", "both"}:
                neighbors.update(self._upstream.get(node, ()))
                neighbors.update(self._merge_peers.get(node, ()))
            if direction in {"downstream", "both"}:
                neighbors.update(self._downstream.get(node, ()))
            if direction == "both":
                for carrier_id in self._carriers_by_object.get(node, ()):
                    neighbors.update(self._carrier_members[carrier_id])
                neighbors.discard(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return tuple(sorted(visited))

    def trace(self, seeds: list[str] | tuple[str, ...], max_depth: int) -> tuple[str, ...]:
        if not seeds:
            raise ValueError("at least one contamination seed is required")
        if len(set(seeds)) != len(seeds):
            raise ValueError("contamination seeds must be unique")
        impacted: set[str] = set()
        for seed in seeds:
            impacted.update(self.related(seed, "both", max_depth))
        return tuple(sorted(impacted))


@dataclass(frozen=True, slots=True)
class QualityHold:
    hold_id: str
    request_id: str
    object_id: str


class HoldBook:
    """Idempotent quality-hold records keyed by recall request."""

    def __init__(self) -> None:
        self._recalls: dict[str, frozenset[str]] = {}
        self._active: dict[str, dict[str, str]] = defaultdict(dict)

    def add_holds(self, request_id: str, object_ids: list[str] | tuple[str, ...]) -> tuple[QualityHold, ...]:
        scope = frozenset(object_ids)
        previous = self._recalls.get(request_id)
        if previous is not None and previous != scope:
            raise ValueError("recall request id reused with a different hold scope")
        self._recalls[request_id] = scope
        added: list[QualityHold] = []
        for object_id in sorted(scope):
            if self._active.get(object_id):
                continue
            hold_id = deterministic_id("HOLD", request_id, object_id)
            self._active[object_id][request_id] = hold_id
            added.append(QualityHold(hold_id, request_id, object_id))
        return tuple(added)

    def active_holds(self, object_id: str) -> tuple[QualityHold, ...]:
        return tuple(
            QualityHold(hold_id, request_id, object_id)
            for request_id, hold_id in sorted(self._active.get(object_id, {}).items())
        )

    def held_objects(self) -> tuple[str, ...]:
        return tuple(sorted(object_id for object_id, holds in self._active.items() if holds))

    def recall_scope(self, request_id: str) -> tuple[str, ...]:
        scope = self._recalls.get(request_id)
        return tuple(sorted(scope)) if scope is not None else ()


class RecallLedger:
    """Stores the original recall result for request-scoped replay."""

    def __init__(self) -> None:
        self._recalls: dict[str, ContaminationRecall] = {}

    def replay(self, request_id: str) -> ContaminationRecall | None:
        return self._recalls.get(request_id)

    def record(self, recall: ContaminationRecall) -> None:
        if recall.request_id in self._recalls:
            raise ValueError("recall result was recorded before replay validation")
        self._recalls[recall.request_id] = recall


@dataclass(frozen=True, slots=True)
class ContaminationRecall:
    request_id: str
    affected_objects: tuple[str, ...]
    held_objects: tuple[str, ...]
    new_holds: tuple[QualityHold, ...]
    shipped_objects: tuple[str, ...]
    disposed_objects: tuple[str, ...]


def recall_contamination(
    request_id: str,
    graph: ContaminationTraceGraph,
    material_states: Mapping[str, MaterialObject],
    seeds: list[str] | tuple[str, ...],
    max_depth: int,
    holds: HoldBook,
    ledger: RecallLedger,
) -> ContaminationRecall:
    previous = ledger.replay(request_id)
    affected = graph.trace(seeds, max_depth)
    if previous is not None and previous.affected_objects != affected:
        raise ValueError("recall request id reused with different contamination scope")

    missing = sorted(object_id for object_id in affected if object_id not in material_states)
    if missing:
        raise ValueError(f"missing material state: {', '.join(missing)}")
    objects = {object_id: material_states[object_id] for object_id in affected}
    if any(object_id != obj.object_id for object_id, obj in objects.items()):
        raise ValueError("material state key does not match object id")

    unknown = sorted(
        object_id for object_id in affected if objects[object_id].state not in set(MaterialState)
    )
    if unknown:
        raise ValueError(f"invalid material state: {', '.join(unknown)}")

    wip = tuple(object_id for object_id in affected if objects[object_id].state == MaterialState.WIP)
    shipped = tuple(object_id for object_id in affected if objects[object_id].state == MaterialState.SHIPPED)
    disposed = tuple(object_id for object_id in affected if objects[object_id].state == MaterialState.DISPOSED)
    if previous is not None:
        held = tuple(object_id for object_id in wip if holds.active_holds(object_id))
        current_values = (held, shipped, disposed)
        previous_values = tuple(getattr(previous, name) for name in ("held_objects", "shipped_objects", "disposed_objects"))
        if current_values != previous_values:
            raise ValueError("recall request id reused with different material state")
        return previous

    new_holds = holds.add_holds(request_id, wip)
    held = tuple(object_id for object_id in wip if holds.active_holds(object_id))
    recall = ContaminationRecall(
        request_id=request_id,
        affected_objects=affected,
        held_objects=held,
        new_holds=new_holds,
        shipped_objects=shipped,
        disposed_objects=disposed,
    )
    ledger.record(recall)
    return recall
