# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import (
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel

from llama_stack.apis.inference import ModelStore
from llama_stack.apis.tools import RAGDocument
from llama_stack.apis.vector_io import Chunk
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ConvertResponse(BaseModel):
    """TODO"""

    converted_by_document_id: dict[str, str] = {}  # Mapped by document_id


@json_schema_type
class ChunkResponse(BaseModel):
    """TODO"""

    document_id: str
    chunks: list[Chunk] = []


@json_schema_type
class DocumentProcessorInput(BaseModel):
    provider_id: str


@runtime_checkable
@trace_protocol
class DocumentProcessors(Protocol):
    @webmethod(route="/document_processors", method="POST")
    async def register_document_processor(
        self,
        provider_id: str,
    ) -> None:
        """TODO"""
        ...

    @webmethod(route="/document_processors", method="GET")
    async def list_document_processors(self) -> list[DocumentProcessorInput]:
        """TODO"""
        ...


@runtime_checkable
@trace_protocol
class DocumentProcessing(Protocol):
    """TODO"""

    model_store: ModelStore

    @webmethod(route="/document-processing/convert", method="POST")
    async def convert(
        self,
        documents: list[RAGDocument],
    ) -> ConvertResponse:
        """TODO"""
        ...

    @webmethod(route="/document-processing/chunk", method="POST")
    async def chunk(
        self,
        # TODO be more specific with the processed document type: where do we set metadata?
        document_id: str,
        content: str,
        window_len: int,
        overlap_len: int,
    ) -> ChunkResponse:
        """TODO"""
        ...
