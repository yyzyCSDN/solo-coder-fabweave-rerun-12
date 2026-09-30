from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping


class RecallGraph:
    """Contamination-spread view over splits, merges and shared carriers.

    Links are (parent, child) genealogy pairs: one parent with many children
    models a split, many parents with one child model a merge.  Carriers map a
    carrier id to the lots it carried.  Contamination crosses every association
    in either direction, so traversal is bidirectional and cycle-safe.
    """

    def __init__(
        self,
        links: Iterable[tuple[str, str]] = (),
        carriers: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._neighbors: dict[str, set[str]] = defaultdict(set)
        for parent, child in links:
            self._connect(parent, child)
        for carrier, members in (carriers or {}).items():
            for member in members:
                self._connect(carrier, member)

    def _connect(self, left: str, right: str) -> None:
        if left == right:
            raise ValueError("self associations are not allowed")
        self._neighbors[left].add(right)
        self._neighbors[right].add(left)

    def related(self, start: str, max_depth: int) -> tuple[str, ...]:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        found: set[str] = set()
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth == max_depth:
                continue
            for neighbor in sorted(self._neighbors.get(node, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                found.add(neighbor)
                queue.append((neighbor, depth + 1))
        return tuple(sorted(found))
