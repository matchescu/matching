from matchescu.extraction import Traits
from matchescu.matching.config import TraitConfig


def get_traits(configs: list[TraitConfig]) -> Traits:
    """Initialize ``Traits`` from ``TraitConfig`` objects.

    :param configs: configs to use

    :return: ``Traits`` instance fully initialized from configs.Ï
    """
    traits = Traits()
    for cfg in configs:
        method = getattr(traits, cfg.type, None)
        if method is None:
            raise ValueError(f"Unknown trait type: '{cfg.type}'")
        traits = method(cfg.keys)
    return traits
