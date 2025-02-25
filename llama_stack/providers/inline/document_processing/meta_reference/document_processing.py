# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.document_processing import DocumentProcessing, ConvertResponse, ChunkResponse
from llama_stack.providers.datatypes import DocumentProcessingProtocolPrivate
from llama_stack.apis.tools import RAGDocument
from llama_stack.apis.vector_io import Chunk
import io
from ..converter import content_from_doc
import httpx


class MetaReferenceDocumentProcessingImpl(DocumentProcessing, DocumentProcessingProtocolPrivate):
    def __init__(
        self,
    ):
        pass

    async def initialize(self) -> None:
        pass

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
        from llama_models.llama3.api.tokenizer import Tokenizer

        tokenizer = Tokenizer.get_instance()
        tokens = tokenizer.encode(content, bos=False, eos=False)

        chunks = []
        for i in range(0, len(tokens), window_len - overlap_len):
            toks = tokens[i : i + window_len]
            chunk = tokenizer.decode(toks)
            # chunk is a string
            chunks.append(
                Chunk(
                    content=chunk,
                    metadata={
                        "token_count": len(toks),
                        "document_id": document_id,
                    },
                )
            )

        return ChunkResponse(
            document_id=document_id,
            chunks=chunks,
        )

    def _parse_pdf(self, data: bytes) -> str:
        from pypdf import PdfReader

        # For PDF and DOC/DOCX files, we can't reliably convert to string
        pdf_bytes = io.BytesIO(data)
        pdf_reader = PdfReader(pdf_bytes)
        return "\n".join([page.extract_text() for page in pdf_reader.pages])

    async def _convert_from_data(self, data: bytes, encoding: str, mime_type: str | None) -> str:
        mime_category = mime_type.split("/")[0]
        if mime_category == "text":
            # For text-based files (including CSV, MD)
            return data.decode(encoding)

        elif mime_type == "application/pdf":
            return self._parse_pdf(data)

        else:
            log.error("Could not extract content from data_url properly.")
            return ""

    async def _convert_from_url(self, data_url: str, mime_type: str | None) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get(data_url)
        if mime_type == "application/pdf":
            return self._parse_pdf(r.content)
        else:
            return r.text
