# DataCat - Data Storage System

[![PyPI version](https://badge.fury.io/py/datacat.svg)](https://pypi.org/project/datacat/)
[![GitHub tag](https://img.shields.io/github/v/tag/papasaidfine/datacat?sort=semver)](https://github.com/papasaidfine/datacat/tags)

A data storage system with catalog storage and pluggable serializers.

## Features

- **CatalogStorage**: Manages DuckDB catalog with hashed file paths
- **Serializer Interface**: Pluggable serialization system
- **SparseMatrixSerializer**: Handles scipy sparse matrices and numpy arrays
- **NumpyArraySerializer**: Pure numpy arrays without pickle dependency

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### With Sparse Matrices

```python
from datacat import CatalogStorage, SparseMatrixSerializer
import numpy as np
import scipy.sparse as sp

# Initialize with sparse matrix support
serializer = SparseMatrixSerializer()
storage = CatalogStorage(
    catalog_columns=['dim1', 'dim2', 'date'],
    serializer=serializer
)

# Save mixed data
data = {
    'returns': sp.csr_matrix([[1, 2, 0], [0, 0, 3]]),
    'stock_names': np.array(['AAPL', 'MSFT']),
    'weights': np.array([0.5, 0.5])
}
storage.save(data, dim1="v1", dim2="v2", date="2024-01-01")
```

### With Pure NumPy Arrays

```python
from datacat import CatalogStorage, NumpyArraySerializer
import numpy as np

# Initialize with numpy-only support (no pickle)
serializer = NumpyArraySerializer()
storage = CatalogStorage(
    catalog_columns=['experiment', 'model', 'date'],
    serializer=serializer
)

# Save pure numpy data
data = {
    'features': np.random.rand(100, 10).astype(np.float32),
    'labels': np.array(['class_A', 'class_B'] * 50),
    'timestamps': np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]')
}
storage.save(data, experiment="classification", model="cnn", date="2024-01-01")
```
