from __future__ import annotations

from fabweave.contamination.recall import (
    CarrierAssociation,
    ContaminationLink,
    ContaminationRecall,
    ContaminationTraceGraph,
    HoldBook,
    LinkKind,
    MaterialObject,
    MaterialState,
    RecallLedger,
    carrier_association,
    merge_link,
    recall_contamination,
    split_link,
)
from fabweave.core.events import EventLog
from fabweave.workflows.control import record_contamination_recall


def _states(**states: MaterialState) -> dict[str, MaterialObject]:
    return {object_id: MaterialObject(object_id, state) for object_id, state in states.items()}


def test_recall_traverses_split_merge_and_carriers_within_bounded_depth():
    graph = ContaminationTraceGraph(
        links=[
            split_link("RAW", "SPLIT-A"),
            split_link("RAW", "SPLIT-B"),
            merge_link(("SPLIT-A", "SPLIT-B"), "MERGED"),
            split_link("MERGED", "DOWNSTREAM"),
        ],
        carrier_associations=[carrier_association("CARRIER-1", ("SPLIT-A", "NEARBY"))],
    )

    assert graph.related("RAW", "downstream", 2) == ("MERGED", "RAW", "SPLIT-A", "SPLIT-B")
    assert graph.trace(["SPLIT-A"], 1) == ("MERGED", "NEARBY", "RAW", "SPLIT-A", "SPLIT-B")
    assert graph.trace(["SPLIT-A"], 2) == (
        "DOWNSTREAM",
        "MERGED",
        "NEARBY",
        "RAW",
        "SPLIT-A",
        "SPLIT-B",
    )


def test_trace_deduplicates_merging_paths_and_survives_cycles():
    graph = ContaminationTraceGraph(
        links=[
            split_link("SEED", "A"),
            split_link("SEED", "B"),
            merge_link(("A", "B"), "MERGED"),
            split_link("MERGED", "A"),
        ],
        carrier_associations=[
            CarrierAssociation("CARRIER-1", ("A", "B")),
            CarrierAssociation("CARRIER-2", ("B", "C")),
            CarrierAssociation("CARRIER-2", ("B", "C")),
        ],
    )

    assert graph.trace(["SEED"], 5) == ("A", "B", "C", "MERGED", "SEED")


def test_recall_holds_wip_and_lists_shipped_or_disposed_separately():
    graph = ContaminationTraceGraph(
        links=[split_link("SEED", "WIP-A"), split_link("WIP-A", "SHIPPED-A")],
        carrier_associations=[carrier_association("CARRIER-1", ("WIP-A", "DISPOSED-A"))],
    )
    states = _states(
        SEED=MaterialState.WIP,
        **{"WIP-A": MaterialState.WIP, "SHIPPED-A": MaterialState.SHIPPED, "DISPOSED-A": MaterialState.DISPOSED},
    )
    holds = HoldBook()
    ledger = RecallLedger()

    recall = recall_contamination("RECALL-1", graph, states, ["SEED"], 3, holds, ledger)

    assert recall.affected_objects == ("DISPOSED-A", "SEED", "SHIPPED-A", "WIP-A")
    assert recall.held_objects == ("SEED", "WIP-A")
    assert {hold.object_id for hold in recall.new_holds} == {"SEED", "WIP-A"}
    assert recall.shipped_objects == ("SHIPPED-A",)
    assert recall.disposed_objects == ("DISPOSED-A",)
    assert holds.held_objects() == ("SEED", "WIP-A")
    assert len(holds.active_holds("WIP-A")) == 1


def test_recall_is_idempotent_and_preserves_existing_overlapping_holds():
    graph = ContaminationTraceGraph(
        links=[split_link("A", "B"), split_link("B", "C")],
    )
    states = _states(A=MaterialState.WIP, B=MaterialState.WIP, C=MaterialState.WIP)
    holds = HoldBook()
    ledger = RecallLedger()

    first = recall_contamination("RECALL-1", graph, states, ["A"], 1, holds, ledger)
    overlapping = recall_contamination("RECALL-2", graph, states, ["B"], 1, holds, ledger)
    replay = recall_contamination("RECALL-1", graph, states, ["A"], 1, holds, ledger)

    assert replay == first
    assert replay.new_holds == first.new_holds
    assert {hold.object_id for hold in overlapping.new_holds} == {"C"}
    assert len(holds.active_holds("B")) == 1
    assert [hold.request_id for hold in holds.active_holds("A")] == ["RECALL-1"]


def test_recorded_recall_is_event_idempotent_and_keeps_non_holdable_objects():
    graph = ContaminationTraceGraph(
        links=[split_link("A", "B")],
        carrier_associations=[carrier_association("CARRIER-1", ("A", "C"))],
    )
    states = _states(A=MaterialState.WIP, B=MaterialState.SHIPPED, C=MaterialState.DISPOSED)
    holds = HoldBook()
    ledger = RecallLedger()
    log = EventLog()

    first = record_contamination_recall(log, "RECALL-1", graph, states, ["A"], 2, holds, ledger, 10)
    replay = record_contamination_recall(log, "RECALL-1", graph, states, ["A"], 2, holds, ledger, 10)
    events = log.read("contamination-recall:RECALL-1")

    assert first == replay
    assert len(events) == 1
    assert events[0].data["held_objects"] == ["A"]
    assert events[0].data["shipped_objects"] == ["B"]
    assert events[0].data["disposed_objects"] == ["C"]


def test_contamination_graph_validates_link_shape_and_known_objects():
    for link in (
        ContaminationLink(LinkKind.SPLIT, ("A", "B"), "C"),
        ContaminationLink(LinkKind.MERGE, ("A",), "B"),
    ):
        try:
            ContaminationTraceGraph(links=[link])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid contamination link was accepted")
