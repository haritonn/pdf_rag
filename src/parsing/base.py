from abc import ABC, abstractmethod


class DocumentParser(ABC):
    """Parsing input file into Document"""

    @abstractmethod
    def parse_file(self, file_path):
        pass
