"""Extract and sample bad triplets from a matching ground truth.

A bad triplet is a triple of records ``(h, b, t)``. The record ``b`` is a
bridge record. Its text looks like the text of records in a target cluster,
but the record belongs to another cluster, called its home cluster. The
record ``h`` is an ordinary member of the home cluster. The record ``t`` is an
ordinary member of the target cluster. The ground truth declares the pair
``(h, b)`` as a match and the pair ``(t, b)`` as a bridge link.

A bridge link points in one direction: ``t`` matches ``b`` in that order,
while ``b`` does not match ``t``. A directed ground truth records that
direction with label 2. A classical binary ground truth can only use label 1.
The sampler therefore does not read labels to find bridges. Instead, the
caller passes a predicate that marks bridge records, and the declared cluster
ground truth separates home pairs from bridge links.

The sampler follows two rules. First, it only uses pairs that the ground
truth declares; it never invents records or pairs. Second, it returns every
triplet that the declared structure allows, in the order of the inputs:
bridges follow their first appearance in the ground truth, and pairs follow
the order of the ground-truth mapping. Exclusions come in two strengths.
Hard exclusions ban exact pairs and never bend, so affected triplets are not
returned. Soft exclusions ban a bridge as a whole and bend under quota
pressure, so affected triplets are returned with ``soft_excluded=True``.
Callers that require strict exclusions filter the marked triplets out;
callers that must reach a target count can fall back to them. The same
inputs always produce the same sample.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.matching._logging import resolve_logger

Pair = tuple[RefId, RefId]
RefKey = tuple[str, str]
UnorderedPair = frozenset[RefId]
Issue = tuple[Pair, str]
Orphan = tuple[RefId, str]


def _ref_key(ref: RefId) -> RefKey:
    return ref.source, str(ref.label)


def _unordered(pair: Pair) -> UnorderedPair:
    return frozenset(pair)


def _counterpart(pair: Pair, ref: RefId) -> RefId:
    left, right = pair
    return right if left == ref else left


class BadTriplet(NamedTuple):
    """A declared ``(h, b, t)`` bad triplet.

    The triplet gives the sampler one stable name for the three records of a
    bridge link. All triplets point at records and pairs that a matcher ground
    truth already declares, so consumers can trust that the structure exists
    in their data.

    :ivar home: Ordinary member of the bridge's home cluster. The pair
        ``(home, bridge)`` is a declared match.
    :ivar bridge: Record whose text resembles the target cluster even though
        the record belongs to the home cluster.
    :ivar target: Ordinary member of the target cluster. The pair
        ``(target, bridge)`` is a declared bridge link.
    :ivar soft_excluded: ``True`` when the triplet comes from a bridge that a
        soft exclusion avoids. The sampler still returns the triplet so that
        callers can fall back to it when the allowed pool is too small.
        Defaults to ``False``.
    """

    home: RefId
    bridge: RefId
    target: RefId
    soft_excluded: bool = False


@dataclass(frozen=True)
class BridgeIndex:
    """Structural bridge records of one matching ground truth.

    The index exists so that sampling does not have to revisit the ground
    truth. It reads every declared matching pair that has a bridge endpoint
    and files the pair by the declared cluster ground truth. A pair inside the
    bridge's home cluster becomes a home pair. A pair across clusters becomes
    a bridge link. The label value is only checked against zero, so binary and
    directed ground truths produce the same index.

    :ivar bridges: records marked as bridges by a predicate.
    :ivar link_pairs: pairs linking bridges to target clusters, in the
        orientation and order of the ground truth.
    :ivar home_pairs: pairs between bridges and members of their home
        cluster, in the orientation and order of the ground truth.
    :ivar issues: pairs kept out of the index, in ground-truth order, along
        with the reason why: ``"bridge_bridge"`` (both ends are bridges),
        ``"unknown_ref"`` (an end is not in the id table) or ``"unclustered"``
        (an end has no declared cluster).
    :ivar orphan_bridges: records marked as bridges that do not form triplets,
        each with its reason: ``"no_links"`` or ``"no_home_pairs"``. The
        existence of orphan bridges should be interpreted as an error.
    """

    bridges: frozenset[RefId]
    link_pairs: Mapping[RefId, tuple[Pair, ...]]
    home_pairs: Mapping[RefId, tuple[Pair, ...]]
    issues: tuple[Issue, ...]
    orphan_bridges: tuple[Orphan, ...]


class ExclusionReport(NamedTuple):
    """Outcome of resolving an exclusion list against a bridge index.

    The report allows users to see which exclusion strength was applied to
    each exclusion. The two strengths behave differently. Hard exclusions ban
    exact pairs and never bend, so triplets that use a hard pair are dropped.
    Soft exclusions ban a bridge as a whole and bend under quota pressure, so
    the triplets of a soft bridge are returned flagged.

    :ivar hard_exclusions: Unordered pairs that no triplet may use. The sampler
        never returns a triplet that uses one of these pairs.
    :ivar avoided_bridges: Bridges that the exclusions avoid. The sampler still
        returns their triplets, flagged as soft-excluded, so that callers can
        fall back to them when the allowed pool is too small.
    :ivar demoted_bridges: Bridges whose bridge-level exclusion was narrowed
        to pair-level ones. A bridge is demoted when its exact exclusion pairs
        still leave declared structure for the bridge.
    :ivar unknown_pairs: Exclusion pairs that were ignored because the ground
        truth does not declare them.
    """

    hard_exclusions: frozenset[UnorderedPair]
    avoided_bridges: frozenset[RefId]
    demoted_bridges: frozenset[RefId]
    unknown_pairs: tuple[Pair, ...]


class BadTripletSample(NamedTuple):
    """A deterministic sample of bad triplets.

    :ivar triplets: Every structurally valid triplet in ground-truth order:
        bridges by their first appearance, then links and home pairs as the
        ground truth lists them. Triplets that an exclusion applies to carry
        ``soft_excluded=True``.
    :ivar exclusions: The exclusion report that resolved the exclusions.
    """

    triplets: tuple[BadTriplet, ...]
    exclusions: ExclusionReport

    @property
    def by_bridge(self) -> Mapping[RefId, tuple[BadTriplet, ...]]:
        """Group the sampled triplets by bridge record.

        :returns: The triplets of each sampled bridge, in the order the
            bridges first appear in the sample. Bridges without sampled
            triplets are absent.
        """
        grouped = defaultdict(list)
        for triplet in self.triplets:
            grouped[triplet.bridge].append(triplet)
        return {bridge: tuple(items) for bridge, items in grouped.items()}


class BadTripletSampler:
    """Sample bad triplets from a matching ground truth.

    The sampler is used in the split generator and the comparison space
    generator. The caller marks bridge records with a predicate instead of
    relying on label values. This keeps the sampler independent of the
    labeling scheme: a dataset can mark bridges by record source, by label 2,
    or by any other rule.

    :param id_table: all entity references in a dataset. A pair that mentions a
        record outside it is reported as unknown and never sampled.
    :param matcher_gt: pairwise ground truth. Only pairs with a non-zero label
        are considered for triplet formation. Home pairs and bridge links are
        part of this ground truth.
    :param cluster_gt: declared cluster of each record. The sampler compares
        cluster numbers to distinguish between home pairs and bridge links.
    :param is_bridge: Predicate that marks bridge records. The sampler runs it
        once for each record in the id table.
    :param log: Logger for sampling diagnostics. Defaults to the package null
        logger, which discards every diagnostic unless the caller supplies a
        logger.
    """

    def __init__(
        self,
        id_table: IdTable,
        matcher_gt: Mapping[Pair, int],
        cluster_gt: Mapping[RefId, int],
        is_bridge: Callable[[RefId], bool],
        log: logging.Logger | None = None,
    ) -> None:
        self.__id_table = id_table
        self.__matcher_gt = matcher_gt
        self.__cluster_gt = cluster_gt
        self.__is_bridge = is_bridge
        self.__log = resolve_logger(log)

    def index(self) -> BridgeIndex:
        """Classify the declared bridge structure.

        The method uses no randomness, so callers can inspect the structure
        before they decide how to sample it. The result depends only on the
        inputs and keeps their order. Repeated calls produce equal indexes.

        :returns: The structural index that backs every sample.
        """
        ref_ids = {ref.id for ref in self.__id_table}
        ordered_bridge_ids = dict.fromkeys(
            ref.id for ref in self.__id_table if self.__is_bridge(ref.id)
        )
        bridge_ids = frozenset(ordered_bridge_ids)
        home, links, issues = self.__classify(ref_ids, bridge_ids)
        index = BridgeIndex(
            bridges=bridge_ids,
            link_pairs=self.__finalized(links),
            home_pairs=self.__finalized(home),
            issues=tuple(issues),
            orphan_bridges=self.__orphan_bridges(ordered_bridge_ids, links, home),
        )
        self.__log.info(
            "bridge index: %d bridges, %d links, %d home pairs, %d issues",
            len(bridge_ids),
            sum(map(len, index.link_pairs.values())),
            sum(map(len, index.home_pairs.values())),
            len(index.issues),
        )
        return index

    def sample(self, exclusions: Iterable[Pair] = ()) -> BadTripletSample:
        """Return every bad triplet that the declared structure allows.

        The method resolves exclusions first and samples afterward. Each
        bridge link ``(t, b)`` combines with every home pair ``(h, b)`` of the
        bridge. The declared structure is the only limit; the method applies
        no cap.

        Exclusions have two strengths. A hard exclusion bans exact pairs and
        never bends, so triplets that use a hard pair are not returned. A soft
        exclusion bans a bridge as a whole and bends under quota pressure, so
        the triplets of a soft bridge are returned with
        ``soft_excluded=True``. Callers that require strict exclusions filter
        the marked triplets out. Callers that must reach a target count can
        fall back to them, which keeps a single pass sufficient for every
        consumer.

        :param exclusions: Pairs to resolve into exclusions. A pair without a
            bridge record becomes a hard exclusion of that pair. A pair with a
            bridge record starts as a soft exclusion of that bridge. The soft
            exclusion is narrowed to hard exclusions of the listed pairs when
            those pairs still leave links and home pairs for the bridge. A
            pair that the ground truth does not declare is reported in
            :attr:`ExclusionReport.unknown_pairs` and otherwise ignored.
            Defaults to no exclusions.
        :returns: The sampled triplets in ground-truth order, together with
            the exclusion report.
        """
        index = self.index()
        report = self.__resolve_exclusions(index, exclusions)
        triplets = []
        for bridge in index.link_pairs:
            triplets.extend(self.__bridge_triplets(index, bridge, report))
        flagged = sum(1 for triplet in triplets if triplet.soft_excluded)
        self.__log.info(
            "sampled %d triplets from %d bridges (%d flagged, %d soft, %d demoted)",
            len(triplets),
            len(index.link_pairs),
            flagged,
            len(report.avoided_bridges),
            len(report.demoted_bridges),
        )
        return BadTripletSample(tuple(triplets), report)

    def __classify(self, refs, bridges):
        home = defaultdict(dict)
        links = defaultdict(dict)
        issues = []
        for pair in self.__matcher_gt:
            if not self.__matcher_gt[pair]:
                continue
            issue = self.__classify_issue(pair, refs, bridges, home, links)
            if issue is not None:
                issues.append((pair, issue))
        return home, links, issues

    def __classify_issue(self, pair, refs, bridges, home, links) -> str | None:
        left, right = pair
        if left not in refs or right not in refs:
            return "unknown_ref"
        bridge_on_left = left in bridges
        bridge_on_right = right in bridges
        if bridge_on_left and bridge_on_right:
            return "bridge_bridge"
        if not (bridge_on_left or bridge_on_right):
            return None
        bridge = left if bridge_on_left else right
        other = _counterpart(pair, bridge)
        bridge_cluster = self.__cluster_gt.get(bridge)
        other_cluster = self.__cluster_gt.get(other)
        if bridge_cluster is None or other_cluster is None:
            return "unclustered"
        bucket = home if bridge_cluster == other_cluster else links
        bucket[bridge].setdefault(_unordered(pair), pair)
        return None

    @staticmethod
    def __finalized(
        bucket: Mapping[RefId, Mapping[UnorderedPair, Pair]],
    ) -> dict[RefId, tuple[Pair, ...]]:
        return {bridge: tuple(pairs.values()) for bridge, pairs in bucket.items()}

    @staticmethod
    def __orphan_bridges(ordered_bridge_ids, links, home) -> tuple[Orphan, ...]:
        orphans = []
        for bridge in ordered_bridge_ids:
            if not links.get(bridge):
                orphans.append((bridge, "no_links"))
            elif not home.get(bridge):
                orphans.append((bridge, "no_home_pairs"))
        return tuple(orphans)

    def __bridge_triplets(self, index, bridge, report) -> list[BadTriplet]:
        flagged = bridge in report.avoided_bridges
        return [
            BadTriplet(
                home=_counterpart(home_pair, bridge),
                bridge=bridge,
                target=_counterpart(link_pair, bridge),
                soft_excluded=flagged,
            )
            for link_pair in index.link_pairs[bridge]
            for home_pair in index.home_pairs.get(bridge, ())
            if self.__admissible(link_pair, home_pair, report)
        ]

    @staticmethod
    def __admissible(link_pair, home_pair, report) -> bool:
        return (
            _unordered(link_pair) not in report.hard_exclusions
            and _unordered(home_pair) not in report.hard_exclusions
        )

    def __resolve_exclusions(self, index, exclusions) -> ExclusionReport:
        hard_pairs, by_bridge, candidates, unknown = self.__partition_exclusions(
            index, exclusions
        )
        soft_bridges, demoted, hard_pairs = self.__cascade(
            index, hard_pairs, by_bridge, candidates
        )
        return ExclusionReport(
            hard_exclusions=frozenset(hard_pairs),
            avoided_bridges=frozenset(soft_bridges),
            demoted_bridges=frozenset(demoted),
            unknown_pairs=tuple(unknown),
        )

    def __partition_exclusions(self, index, exclusions):
        refs = {ref.id for ref in self.__id_table}
        hard_pairs: set[UnorderedPair] = set()
        by_bridge: dict[RefId, set[UnorderedPair]] = defaultdict(set)
        candidates: set[RefId] = set()
        unknown: list[Pair] = []
        for pair in exclusions:
            pair = (pair[0], pair[1])
            if self.__is_unknown(pair, refs):
                unknown.append(pair)
                continue
            bridges = self.__bridge_ends(pair, index)
            if not bridges:
                hard_pairs.add(_unordered(pair))
                continue
            candidates.update(bridges)
            for bridge in bridges:
                by_bridge[bridge].add(_unordered(pair))
        return hard_pairs, by_bridge, candidates, unknown

    def __is_unknown(self, pair: Pair, refs) -> bool:
        left, right = pair
        if left not in refs or right not in refs:
            return True
        return pair not in self.__matcher_gt and (right, left) not in self.__matcher_gt

    @staticmethod
    def __bridge_ends(pair: Pair, index: BridgeIndex) -> list[RefId]:
        return sorted({ref for ref in pair if ref in index.bridges}, key=_ref_key)

    def __cascade(self, index, hard_pairs, by_bridge, candidates):
        working = set(hard_pairs)
        demoted = set()
        for bridge in sorted(candidates, key=_ref_key):
            trial = working | by_bridge[bridge]
            if self.__has_structure(index, bridge, trial):
                working = trial
                demoted.add(bridge)
        return candidates - demoted, demoted, working

    def __has_structure(self, index, bridge, hard_pairs) -> bool:
        links = index.link_pairs.get(bridge, ())
        homes = index.home_pairs.get(bridge, ())
        return self.__any_usable(links, hard_pairs) and self.__any_usable(
            homes, hard_pairs
        )

    @staticmethod
    def __any_usable(pairs, hard_pairs) -> bool:
        return any(_unordered(pair) not in hard_pairs for pair in pairs)
