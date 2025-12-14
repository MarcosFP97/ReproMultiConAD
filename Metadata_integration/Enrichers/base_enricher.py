from abc import ABC, abstractmethod


class BaseMetadataEnricher(ABC):
    """
    Base class for dataset-specific metadata enrichers.

    Each enricher adds missing metadata fields to a parsed CHA
    data point according to the rules of a specific dataset.
    """

    @abstractmethod
    def enrich(self, info: dict) -> dict:
        """
        Enrich a parsed CHA metadata dictionary in a dataset-specific way.

        Parameters
        ----------
        info : dict
            Parsed metadata from a .cha file.

        Returns
        -------
        dict
            The enriched metadata dictionary.
        """
        pass

