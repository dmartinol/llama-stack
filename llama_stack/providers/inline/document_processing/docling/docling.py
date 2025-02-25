# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0

# This is the implementation code for the ilab rag convert command, which converts documents from their
# native format (e.g., PDF) to Docling JSON for use by ilab rag ingest.  See also the code in cli/rag/convert.py
# which instantiates the CLI command and calls out to the methods in this file.

import json
import logging
import tempfile
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat  # type: ignore
from docling.datamodel.document import ConversionResult  # type: ignore
from docling.datamodel.pipeline_options import (  # type: ignore
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,  # type: ignore
    PdfFormatOption,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument
from pydantic_core._pydantic_core import ValidationError

from llama_stack.apis.vector_io import Chunk

from .config import DoclingImplConfig
from llama_stack.providers.datatypes import DocumentProcessingProtocolPrivate
from llama_stack.apis.document_processing import (
    DocumentProcessing,
    ConvertResponse,
    ChunkResponse,
)
from ..converter import content_from_doc
from llama_stack.apis.tools import RAGDocument

logger = logging.getLogger(__name__)


class DoclingDocumentProcessingImpl(DocumentProcessing, DocumentProcessingProtocolPrivate):
    def __init__(self, config: DoclingImplConfig) -> None:
        if config.tokenizer_model_id is None:
            raise ValueError("Missing `tokenizer_model_id` in docling configuration")
        self.export_type = config.export_type

        self.config = config
        self._document_converter: DocumentConverter | None = None
        self._chunker: HybridChunker | None = None

    async def initialize(self) -> None:
        pass

    @property
    def document_converter(self) -> DocumentConverter:
        if self._document_converter is None:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.AUTO, **(self.config.accelerator_options or {})
            )
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                accelerator_options=accelerator_options,
            )
            self._document_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        return self._document_converter

    @property
    def chunker(self) -> HybridChunker:
        if self._chunker is None:
            # TODO: use configured model-id instead
            self._chunker = HybridChunker(
                tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            )
        return self._chunker

    async def convert(
        self,
        documents: list[RAGDocument],
    ) -> ConvertResponse:
        response = ConvertResponse()
        for doc in documents:
            response.converted_by_document_id[doc.document_id] = await content_from_doc(
                doc, self._convert_from_data, self._convert_from_url
            )
        return response

    async def chunk(self, document_id: str, content: str, window_len: int, overlap_len: int) -> ChunkResponse:
        try:
            document = DoclingDocument.model_validate_json(content)
            chunks = []
            for chunk in self.chunker.chunk(document):
                chunks.append(
                    Chunk(
                        content=chunk.text,
                        metadata={
                            "token_count": len(chunk.text),  # TBD: text len is not reflecting # of tokens
                            "document_id": document_id,
                        },
                        # Todo adapt chunk.meta instead docling_core.transforms.chunker.DocMeta
                    )
                )
                return ChunkResponse(
                    document_id=document_id,
                    chunks=chunks,
                )
        except ValidationError as e:
            logger.error(f"Expected {file_path} to be in docling format, but schema validation failed: {e}")
            raise e

    async def _convert_from_data(self, data: bytes, encoding: str, mime_type: str | None) -> str:
        file_type = mime_type.split("/")[1]  # Guessing
        with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as tmp_pdf:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            logger.info(f"Saved as {tmp_file}")

            result = self.document_converter.convert(tmp_file, raises_on_error=False)
            return result.document.export_to_dict()

    async def _convert_from_url(self, data_url: str, mime_type: str | None) -> str:
        result = self.document_converter.convert(data_url, raises_on_error=False)
        # TODO make the output format configurable
        if self.export_type == "JSON":
            result_text = json.dumps(result.document.export_to_dict())
        else:
            result_text = result.document.export_to_markdown()

        # TO REMOVE
        # ADD PARAMETER TO  CONFIGURE THIS STEP
        _export_documents(result, Path("./converted_docs"))
        return result_text


def _export_documents(
    conversion_result: ConversionResult,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    if conversion_result.status == ConversionStatus.SUCCESS:
        success_count += 1
        doc_filename = conversion_result.input.file.stem

        with (output_dir / f"{doc_filename}.json").open("w") as fp:
            fp.write(json.dumps(conversion_result.document.export_to_dict()))
        with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
            fp.write(conversion_result.document.export_to_document_tokens())
    elif conversion_result.status == ConversionStatus.PARTIAL_SUCCESS:
        logger.info(f"Document {conversion_result.input.file} was partially converted with the following errors:")
        for item in conversion_result.errors:
            print(f"\t{item.error_message}")
        partial_success_count += 1
    else:
        logger.info(f"Document {conversion_result.input.file} failed to convert.")
        failure_count += 1

    logger.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count
