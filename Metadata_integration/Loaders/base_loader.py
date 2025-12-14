from abc import ABC, abstractmethod


class BaseMetadataLoader(ABC):
    """
    Base class for metadata loaders.
    Each loader must implement load_metadata()
    and return a dictionary indexed by file_id.
    """

    @abstractmethod
    def load_metadata(self):
        """
        Load and return metadata as a dictionary.

        Returns
        -------
        dict
            {
                file_id (str): {
                    "Age": int or str,
                    "Gender": str,
                    "Diagnosis": str,
                    ...
                }
            }
        """
        pass
