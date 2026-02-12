# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "bm25_retriever": ["InMemoryBM25Retriever"],
    "dat_hybrid_retriever": ["InMemoryDATHybridRetriever"],
    "embedding_retriever": ["InMemoryEmbeddingRetriever"],
}

if TYPE_CHECKING:
    from .bm25_retriever import InMemoryBM25Retriever as InMemoryBM25Retriever
    from .dat_hybrid_retriever import InMemoryDATHybridRetriever as InMemoryDATHybridRetriever
    from .embedding_retriever import InMemoryEmbeddingRetriever as InMemoryEmbeddingRetriever

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
