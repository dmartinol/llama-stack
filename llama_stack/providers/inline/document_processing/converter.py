# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.tools import RAGDocument
from llama_stack.apis.common.content_types import URL
import re


async def content_from_doc(doc: RAGDocument, convert_from_data_fn, convert_from_url_fn) -> str:
    doc_url: str | None = None
    if isinstance(doc.content, URL):
        doc_url = doc.content.uri
    else:
        pattern = re.compile("^(https?://|file://|data:)")
        if pattern.match(doc.content):
            doc_url = doc.content

    if doc_url:
        if doc_url.startswith("data:"):
            data, encoding, mime_type = self._decode_data(doc_url)
            # TODO probably not working with docling
            # Need to save the file first then use the converter
            return await convert_from_data_fn(data, encoding, mime_type)
        else:
            return await convert_from_url_fn(doc_url, doc.mime_type)

    return interleaved_content_as_str(doc.content)


def _decode_data(self, data_url: str) -> (str, str, str):
    parts = parse_data_url(data_url)
    data = parts["data"]

    if parts["is_base64"]:
        data = base64.b64decode(data)
    else:
        data = unquote(data)
        encoding = parts["encoding"] or "utf-8"
        data = data.encode(encoding)

    encoding = parts["encoding"]
    if not encoding:
        detected = chardet.detect(data)
        encoding = detected["encoding"]

    mime_type = parts["mimetype"]
    return data, encoding, mime_type
