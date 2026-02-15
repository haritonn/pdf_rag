from abc import ABC, abstractmethod
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from ..models.document import Document, Element
# import pandas as pd


class DocumentParser(ABC):
    """Parsing input file into Document"""

    @abstractmethod
    def parse_file(self, file_path):
        pass


class DoclingParser(DocumentParser):
    def __init__(self, extract_images=False, extract_tables=False):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.extract_tables = extract_tables
        self.extract_images = extract_images

    def parse_file(self, file_path):
        result = self.converter.converrt(file_path)
        doc = result.document

        elements = []

        for item in doc.texts:
            element_type = self._get_element_type(item)

            elements.append(
                Element(
                    type=element_type,
                    content=item.text,
                    metadata={
                        "label": item.label,
                        "page": item.prov[0].page_no if item.prov else None,
                    },
                )
            )

        for i, table in enumerate(doc.tables):
            df = table.export_to_dataframe(doc=doc)
            elements.append(
                Element(
                    type="table",
                    content=df.to_markdown(index=False),
                    metadata={
                        "table_id": i,
                        "caption": getattr(table, "caption_text", None),
                        "page": table.prov[0].page_no if table.prov else None,
                    },
                )
            )

        return Document(
            elements=elements,
            metadata={"file_name": Path(file_path).name, "num_pages": len(doc.pages)},
        )

    def _get_element_type(self, element):
        label = element.label.lower()

        if "code" in label:
            return "code"
        elif "formula" in label or "equation" in label:
            return "formula"
        else:
            return "text"
