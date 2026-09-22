from abc import ABC, abstractmethod


class BaseMetadataEnricher(ABC):
    """
    Base class for enriching dataset-specific metadata.

    Each enricher adds missing metadata to the base JSONL created for the collection.
    """

    @abstractmethod
    def enrich(self, info: dict) -> dict:
        """
        Enrich the JSONL entry associated with the dataset.

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

