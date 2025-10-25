from typing import Callable, Optional

try:
    from ppjoin import ppjoin
except ImportError:

    class ppjoin(object):
        @staticmethod
        def whitespace_tokenizer(_):
            return []

        @staticmethod
        def join(_, __):
            return []


from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier, EntityReference


class PPJoin(object):
    def __init__(
        self,
        threshold: float,
        ref_flattener: Optional[Callable[[EntityReference], str]] = None,
    ) -> None:
        if not isinstance(threshold, float) or 0 > threshold or threshold > 1:
            raise ValueError(f"'{threshold}' is not a valid Jaccard threshold")
        self.__t = threshold
        self.__to_str = ref_flattener or self._default_reference_flattener

    @staticmethod
    def _default_reference_flattener(ref: EntityReference) -> str:
        return " ".join(
            str(item).lower().casefold() for item in ref if item is not None
        )

    def _tokenizer(self, reference: EntityReference) -> set[str]:
        return set(ppjoin.whitespace_tokenizer(self.__to_str(reference)))

    def predict(
        self, id_pairs: BinaryComparisonSpace, id_table: IdTable
    ) -> set[tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]]:
        refs = list(map(id_table.get_all, id_pairs))
        tokenized_refs = list(
            (self._tokenizer(e1), self._tokenizer(e2)) for e1, e2 in refs
        )
        datasets = list(map(list, zip(*tokenized_refs)))
        ppjoin_result = ppjoin.join(datasets, self.__t)
        id_pairs_check = set(id_pairs)
        result = set(
            id_pair
            for (l_src, l_id), (r_src, r_id) in ppjoin_result
            if (id_pair := (refs[l_id][l_src].id, refs[r_id][r_src].id))
            in id_pairs_check
        )
        return result
