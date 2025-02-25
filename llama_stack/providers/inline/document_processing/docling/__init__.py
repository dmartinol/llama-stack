# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import DoclingImplConfig


async def get_provider_impl(config: DoclingImplConfig, deps: Dict[Api, ProviderSpec]):
    from .docling import (
        DoclingDocumentProcessingImpl,
    )

    impl = DoclingDocumentProcessingImpl(config)
    await impl.initialize()
    return impl
