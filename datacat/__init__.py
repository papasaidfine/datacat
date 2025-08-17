"""
DataCat - Data Storage System with Catalog Storage and Pluggable Serializers
"""

from .catalog_storage import CatalogStorage
from .serializers import Serializer, SparseMatrixSerializer, NumpyArraySerializer, AutoSerializer

__version__ = "1.0.0"
__all__ = ["CatalogStorage", "Serializer", "SparseMatrixSerializer", "NumpyArraySerializer", "AutoSerializer"]
