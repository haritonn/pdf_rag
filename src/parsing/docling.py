from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from ..models.document import Document, Element
from .base import DocumentParser

from pathlib import Path


class DoclingParser(DocumentParser):
    def __init__(self, extract_images=False, extract_tables=True):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = extract_tables
        pipeline_options.generate_picture_images = extract_images
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.extract_tables = extract_tables
        self.extract_images = extract_images

    def parse_file(self, file_path):
        result = self.converter.convert(file_path)
        doc = result.document

        elements = []

        for item in doc.texts:
            bbox = None
            if item.prov:
                bbox = {
                    "x0": item.prov[0].bbox.l,
                    "y0": item.prov[0].bbox.t,
                    "x1": item.prov[0].bbox.r,
                    "y1": item.prov[0].bbox.b,
                }

            elements.append(
                Element(
                    type=self._get_element_type(item),
                    content=item.text,
                    metadata={
                        "label": item.label,
                        "page": item.prov[0].page_no if item.prov else None,
                        "bbox": bbox,
                        "heading_level": self._get_heading_level(item.label),
                        "is_list_item": "list" in item.label.lower(),
                        "char_count": len(item.text),
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
        if self.extract_images:
            images_dir = Path("images")
            images_dir.mkdir(exists_ok=True, parents=True)
            for i, pic in enumerate(doc.pictures):
                try:
                    pil_image = None
                    if hasattr(pic.image, "pil_image"):
                        pil_image = pic.image.pil_image

                    caption = getattr(pic, "caption_text", None) or ""
                    classification = None
                    if hasattr(pic, "annotations") and pic.annotations:
                        classification = pic.annotations[0].label

                    image_path = None
                    if pil_image:
                        image_path = f"images/{Path(file_path).stem}_{i}.png"
                        pil_image.save(image_path)

                    elements.append(
                        Element(
                            type="image",
                            content=caption,
                            metadata={
                                "image_id": i,
                                "image_path": Path(image_path),
                                "classification": classification,
                                "page": pic.prov[0].page_no if pic.prov else None,
                                "bbox": (
                                    pic.prov[0].bbox.as_tuple() if pic.prov else None
                                ),
                            },
                        )
                    )
                except Exception as e:
                    print(f"Warning: Could not extract image {i}: {e}")

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

    def _get_heading_level(self, label):
        """Heading level"""
        label_lower = label.lower()

        if "title" in label_lower:
            return 1
        elif "section_header" in label_lower or "section-header" in label_lower:
            return 2
        elif "subsection" in label_lower:
            return 3
        elif "caption" in label_lower:
            return 4

        return None
