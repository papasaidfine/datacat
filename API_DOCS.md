# DataCat API Documentation

## Overview

DataCat is a data storage system that provides efficient storage and retrieval of scientific data (sparse matrices and numpy arrays) with metadata cataloging using DuckDB.

## Core Components

### 1. Serializer Interface

The `Serializer` abstract base class defines the contract for all data serializers:

```python
from abc import ABC, abstractmethod

class Serializer(ABC):
    @abstractmethod
    def save(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """Save data dictionary to file."""
        pass
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load data from file."""
        pass
    
    @abstractmethod
    def delete(self, file_path: Union[str, Path]) -> None:
        """Delete file."""
        pass
    
    @abstractmethod
    def update(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """Update existing file with new data."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension used by this serializer."""
        pass
```

### 2. SparseMatrixSerializer

A concrete implementation that handles scipy sparse matrices and numpy arrays:

**Features:**
- Supports all scipy sparse matrix formats (CSR, CSC, COO, etc.)
- Handles numpy arrays (numeric and string)
- Preserves data types and formats
- Uses .npz compression for efficiency

**Example:**
```python
from datacat import SparseMatrixSerializer
import scipy.sparse as sp
import numpy as np

serializer = SparseMatrixSerializer()

# Mixed data
data = {
    'sparse_matrix': sp.csr_matrix([[1, 0, 2], [0, 3, 0]]),
    'dense_array': np.array([1.5, 2.5, 3.5]),
    'string_labels': np.array(['A', 'B', 'C'])
}

serializer.save("data.npz", data)
loaded = serializer.load("data.npz")
```

### 3. NumpyArraySerializer

A concrete implementation for pure numpy arrays without pickle dependency:

**Features:**
- Handles all numpy array types (numeric, string, boolean, datetime, etc.)
- Uses individual .npy files (no pickle dependency)
- Preserves exact dtypes and shapes
- Directory-based storage allows selective loading
- Does NOT support scipy sparse matrices

**Example:**
```python
from datacat import NumpyArraySerializer
import numpy as np

serializer = NumpyArraySerializer()

# Pure numpy data
data = {
    'features': np.random.rand(100, 10).astype(np.float32),
    'labels': np.array(['class_A', 'class_B'] * 50),
    'timestamps': np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]'),
    'flags': np.array([True, False, True])
}

serializer.save("data_dir", data)  # Creates directory with .npy files
loaded = serializer.load("data_dir")
```

### 4. CatalogStorage

The main storage system that coordinates between cataloging and serialization:

**Features:**
- DuckDB-based metadata catalog
- SHA256-based hashed file paths
- Automatic directory structure: `data/{hash[:2]}/{hash[2:4]}/{full_hash}.ext`
- Consistent catalog and file operations
- Flexible querying and filtering

**Initialization:**
```python
from datacat import CatalogStorage, SparseMatrixSerializer

serializer = SparseMatrixSerializer()
storage = CatalogStorage(
    catalog_columns=['dim1', 'dim2', 'date'],
    serializer=serializer,
    catalog_db_path="my_catalog.db",    # Optional
    data_root="my_data"                 # Optional
)
```

## Core Operations

### Saving Data

```python
data = {
    'returns': sparse_matrix,
    'stock_names': np.array(['AAPL', 'MSFT']),
    'weights': np.array([0.5, 0.5])
}

# Save with metadata - metadata must match catalog_columns exactly
hash_id = storage.save(data, dim1="portfolio", dim2="tech", date="2024-01-01")
```

### Loading Data

```python
# Load by hash ID
loaded_data = storage.load(hash_id)
print(loaded_data['stock_names'])  # ['AAPL', 'MSFT']
```

### Querying Catalog

```python
# List all entries
all_entries = storage.list_all()

# Filter by metadata
tech_entries = storage.query(dim2="tech")
recent_entries = storage.query(date="2024-01-01")

# Multiple filters
specific = storage.query(dim1="portfolio", dim2="tech")

# Custom SQL where clause
custom = storage.query(where_clause="date > '2024-01-01'", order_by="date")

# Limit results
latest = storage.query(order_by="created_at DESC", limit=10)
```

### Updating Data

```python
# Update data and/or metadata
update_data = {'new_field': np.array([1, 2, 3])}
storage.update(hash_id, update_data, date="2024-01-02")
```

### Deleting Data

```python
# Delete data and catalog entry
storage.delete(hash_id)
```

### Statistics

```python
stats = storage.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Column stats: {stats['column_stats']}")
```

## Advanced Usage

### Choosing the Right Serializer

**SparseMatrixSerializer** - Use when you have:
- Mixed sparse matrices and numpy arrays
- Need compressed storage (.npz format)
- Want all data in a single file
- Working with scipy sparse data

**NumpyArraySerializer** - Use when you have:
- Pure numpy arrays only
- Want to avoid pickle dependency
- Need individual file access to arrays
- Want maximum compatibility and security
- Working with large datasets where you might want to load only specific arrays

Example comparison:
```python
# For mixed sparse/dense data
sparse_serializer = SparseMatrixSerializer()
sparse_storage = CatalogStorage(['dim1', 'dim2'], sparse_serializer)

mixed_data = {
    'sparse_matrix': sp.csr_matrix([[1, 0, 2]]),
    'dense_array': np.array([1, 2, 3])
}
sparse_storage.save(mixed_data, dim1="test", dim2="mixed")

# For pure numpy data
numpy_serializer = NumpyArraySerializer() 
numpy_storage = CatalogStorage(['experiment', 'date'], numpy_serializer)

numpy_data = {
    'features': np.random.rand(100, 10),
    'labels': np.array(['A', 'B'] * 50),
    'metadata': np.array(['info1', 'info2'])
}
numpy_storage.save(numpy_data, experiment="test", date="2024-01-01")
```

### Custom Serializers

You can create custom serializers by implementing the `Serializer` interface:

```python
class JSONSerializer(Serializer):
    @property
    def file_extension(self) -> str:
        return ".json"
    
    def save(self, file_path, data_dict):
        with open(file_path, 'w') as f:
            json.dump(data_dict, f)
    
    def load(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def delete(self, file_path):
        Path(file_path).unlink()
    
    def update(self, file_path, data_dict):
        if Path(file_path).exists():
            existing = self.load(file_path)
            existing.update(data_dict)
            self.save(file_path, existing)
        else:
            self.save(file_path, data_dict)

# Use with CatalogStorage
json_serializer = JSONSerializer()
storage = CatalogStorage(['key1', 'key2'], json_serializer)
```

### Hash-based Deduplication

DataCat automatically handles deduplication based on metadata. If you save data with identical metadata, it will overwrite the previous entry:

```python
# These will have the same hash_id
hash1 = storage.save(data1, dim1="A", dim2="B", date="2024-01-01")
hash2 = storage.save(data2, dim1="A", dim2="B", date="2024-01-01")  # Overwrites data1

assert hash1 == hash2
```

### Directory Structure

DataCat creates a hierarchical directory structure based on hash values:

```
data/
├── b9/
│   └── 19/
│       └── b919f4eb77badffe4054153d33556193fc042f261bf2d7f3acb05a77a6c94208.npz
├── 49/
│   └── c7/
│       └── 49c768deaa72008dfbabdee4a9ec9b377bdf2f62cb913b24ed462317d33e36ae.npz
└── ...
```

This structure:
- Prevents filesystem limitations on files per directory
- Enables efficient file system operations
- Maintains consistent performance as data grows

## Error Handling

DataCat provides clear error messages for common issues:

```python
# Invalid metadata schema
try:
    storage.save(data, dim1="A")  # Missing dim2 and date
except ValueError as e:
    print(e)  # "Metadata keys don't match catalog columns"

# Hash ID not found
try:
    storage.load("nonexistent_hash")
except ValueError as e:
    print(e)  # "Hash ID not found in catalog"

# File not found
try:
    serializer.load("nonexistent.npz")
except FileNotFoundError as e:
    print(e)  # "File not found: nonexistent.npz"
```

## Performance Considerations

1. **Sparse Matrix Storage**: All sparse matrices are converted to CSR format for storage, then converted back to original format on load. This ensures consistent storage while preserving format information.

2. **Compression**: Uses numpy's `savez_compressed` for efficient storage of numerical data.

3. **Indexing**: DuckDB provides efficient querying and indexing of catalog metadata.

4. **Hash Collisions**: SHA256 provides virtually no collision risk for practical datasets.

## Best Practices

1. **Consistent Metadata Schema**: Define your catalog columns upfront and stick to them throughout your project.

2. **Meaningful Metadata**: Use metadata that helps you find and organize your data later.

3. **Batch Operations**: For multiple related datasets, consider using consistent metadata patterns for easy querying.

4. **Backup**: Both the catalog database and data directory should be included in backups.

5. **Version Control**: Consider adding version information to your metadata schema for data versioning.

Example with versioning:
```python
storage = CatalogStorage(
    catalog_columns=['dataset_name', 'version', 'date', 'author'],
    serializer=serializer
)

storage.save(data, 
    dataset_name="market_data",
    version="v1.0",
    date="2024-01-01",
    author="analyst1"
)
```
