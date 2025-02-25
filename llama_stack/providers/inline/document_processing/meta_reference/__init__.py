# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


async def get_provider_impl(
    _config: any,
):
    from .document_processing import MetaReferenceDocumentProcessingImpl

    impl = MetaReferenceDocumentProcessingImpl()
    await impl.initialize()
    return impl
