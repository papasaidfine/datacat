"""
Catalog storage system using DuckDB for metadata management.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import duckdb
from .serializers import Serializer


class CatalogStorage:
    """
    Manages DuckDB catalog, hashed file paths, and coordinates operations with data serializer.
    
    This class provides:
    - DuckDB-based catalog management
    - SHA256-based hashed file paths
    - Delegation of data operations to pluggable serializers
    - Catalog consistency with file operations
    - Flexible catalog querying and filtering
    """
    
    def __init__(self, 
                 catalog_columns: List[str], 
                 serializer: Serializer,
                 catalog_db_path: Optional[Union[str, Path]] = None,
                 data_root: Optional[Union[str, Path]] = None):
        """
        Initialize the catalog storage system.
        
        Args:
            catalog_columns: List of column names for the catalog schema
            serializer: Serializer object to handle data operations
            catalog_db_path: Path to DuckDB catalog file (default: ./catalog.db)
            data_root: Root directory for data files (default: ./data)
        """
        self.catalog_columns = catalog_columns
        self.serializer = serializer
        self.catalog_db_path = Path(catalog_db_path or "catalog.db")
        self.data_root = Path(data_root or "data")
        
        # Create data root directory
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection and catalog table
        self._init_catalog()
    
    def _init_catalog(self) -> None:
        """Initialize the DuckDB catalog table with the specified schema."""
        # Create catalog table if it doesn't exist
        columns_sql = ", ".join([f"{col} VARCHAR" for col in self.catalog_columns])
        
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS catalog (
                    hash_id VARCHAR PRIMARY KEY,
                    file_path VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    {columns_sql}
                )
            """)
    
    def _generate_hash_path(self, **kwargs) -> tuple[str, Path]:
        """
        Generate SHA256-based hashed file path.
        
        Args:
            **kwargs: Catalog metadata for hashing
            
        Returns:
            Tuple of (hash_id, full_file_path)
        """
        # Create deterministic hash from sorted metadata
        sorted_items = sorted(kwargs.items())
        hash_input = json.dumps(sorted_items, sort_keys=True)
        hash_id = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Generate path: data/{hash[:2]}/{hash[2:4]}/{full_hash}.ext
        hash_path = self.data_root / hash_id[:2] / hash_id[2:4] / f"{hash_id}{self.serializer.file_extension}"
        
        return hash_id, hash_path
    
    def save(self, data: Dict[str, Any], **metadata) -> str:
        """
        Save data with metadata to the catalog and file system.
        
        Args:
            data: Dictionary containing data to save
            **metadata: Catalog metadata (must match catalog_columns)
            
        Returns:
            Hash ID of the saved data
            
        Raises:
            ValueError: If metadata doesn't match catalog schema
        """
        # Validate metadata matches catalog schema
        if set(metadata.keys()) != set(self.catalog_columns):
            raise ValueError(f"Metadata keys {list(metadata.keys())} don't match catalog columns {self.catalog_columns}")
        
        # Generate hash and file path
        hash_id, file_path = self._generate_hash_path(**metadata)
        
        # Save data using serializer
        self.serializer.save(file_path, data)
        
        # Update catalog
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            # Check if entry already exists
            existing = conn.execute(
                "SELECT hash_id FROM catalog WHERE hash_id = ?", 
                (hash_id,)
            ).fetchone()
            
            if existing:
                # Update existing entry
                set_clause = ", ".join([f"{col} = ?" for col in self.catalog_columns])
                values = [metadata[col] for col in self.catalog_columns] + [hash_id]
                
                conn.execute(f"""
                    UPDATE catalog 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
                    WHERE hash_id = ?
                """, values)
            else:
                # Insert new entry
                columns = ["hash_id", "file_path"] + self.catalog_columns
                placeholders = ", ".join(["?"] * len(columns))
                values = [hash_id, str(file_path)] + [metadata[col] for col in self.catalog_columns]
                
                conn.execute(f"""
                    INSERT INTO catalog ({", ".join(columns)}) 
                    VALUES ({placeholders})
                """, values)
        
        return hash_id
    
    def load(self, hash_id: str) -> Dict[str, Any]:
        """
        Load data by hash ID.
        
        Args:
            hash_id: Hash ID of the data to load
            
        Returns:
            Dictionary containing the loaded data
            
        Raises:
            ValueError: If hash_id not found in catalog
        """
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            result = conn.execute(
                "SELECT file_path FROM catalog WHERE hash_id = ?", 
                (hash_id,)
            ).fetchone()
            
            if not result:
                raise ValueError(f"Hash ID {hash_id} not found in catalog")
            
            file_path = Path(result[0])
        
        return self.serializer.load(file_path)
    
    def delete(self, hash_id: str) -> None:
        """
        Delete data by hash ID.
        
        Args:
            hash_id: Hash ID of the data to delete
            
        Raises:
            ValueError: If hash_id not found in catalog
        """
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            result = conn.execute(
                "SELECT file_path FROM catalog WHERE hash_id = ?", 
                (hash_id,)
            ).fetchone()
            
            if not result:
                raise ValueError(f"Hash ID {hash_id} not found in catalog")
            
            file_path = Path(result[0])
            
            # Delete file using serializer
            self.serializer.delete(file_path)
            
            # Remove from catalog
            conn.execute("DELETE FROM catalog WHERE hash_id = ?", (hash_id,))
    
    def update(self, hash_id: str, data: Dict[str, Any], **metadata) -> None:
        """
        Update existing data by hash ID.
        
        Args:
            hash_id: Hash ID of the data to update
            data: New data dictionary
            **metadata: Updated catalog metadata (optional)
        
        Raises:
            ValueError: If hash_id not found in catalog or metadata schema mismatch
        """
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            result = conn.execute(
                "SELECT file_path FROM catalog WHERE hash_id = ?", 
                (hash_id,)
            ).fetchone()
            
            if not result:
                raise ValueError(f"Hash ID {hash_id} not found in catalog")
            
            file_path = Path(result[0])
        
        # Update data using serializer
        self.serializer.update(file_path, data)
        
        # Update catalog metadata if provided
        if metadata:
            # Validate metadata
            invalid_keys = set(metadata.keys()) - set(self.catalog_columns)
            if invalid_keys:
                raise ValueError(f"Invalid metadata keys: {invalid_keys}. Valid keys: {self.catalog_columns}")
            
            set_clause = ", ".join([f"{col} = ?" for col in metadata.keys()])
            values = list(metadata.values()) + [hash_id]
            
            with duckdb.connect(str(self.catalog_db_path)) as conn:
                conn.execute(f"""
                    UPDATE catalog 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
                    WHERE hash_id = ?
                """, values)
    
    def query(self, 
              where_clause: Optional[str] = None, 
              order_by: Optional[str] = None,
              limit: Optional[int] = None,
              **filters) -> List[Dict[str, Any]]:
        """
        Query the catalog with flexible filtering.
        
        Args:
            where_clause: Raw SQL WHERE clause (optional)
            order_by: Column name to order by (optional)
            limit: Maximum number of results (optional)
            **filters: Column=value filters for exact matches
            
        Returns:
            List of dictionaries containing catalog entries
        """
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            query_parts = ["SELECT * FROM catalog"]
            params = []
            
            # Build WHERE clause
            where_conditions = []
            
            # Add filter conditions
            if filters:
                for col, value in filters.items():
                    if col not in self.catalog_columns + ['hash_id', 'file_path']:
                        raise ValueError(f"Unknown filter column: {col}")
                    where_conditions.append(f"{col} = ?")
                    params.append(value)
            
            # Add custom where clause
            if where_clause:
                where_conditions.append(f"({where_clause})")
            
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))
            
            # Add ORDER BY
            if order_by:
                if order_by not in self.catalog_columns + ['hash_id', 'file_path', 'created_at', 'updated_at']:
                    raise ValueError(f"Unknown order_by column: {order_by}")
                query_parts.append(f"ORDER BY {order_by}")
            
            # Add LIMIT
            if limit:
                query_parts.append(f"LIMIT {limit}")
            
            sql = " ".join(query_parts)
            
            result = conn.execute(sql, params).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            return [dict(zip(columns, row)) for row in result]
    
    def list_all(self, order_by: Optional[str] = "created_at") -> List[Dict[str, Any]]:
        """
        List all catalog entries.
        
        Args:
            order_by: Column to order by (default: created_at)
            
        Returns:
            List of all catalog entries
        """
        return self.query(order_by=order_by)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get catalog statistics.
        
        Returns:
            Dictionary with catalog statistics
        """
        with duckdb.connect(str(self.catalog_db_path)) as conn:
            total_count = conn.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
            
            # Get column value counts
            column_stats = {}
            for col in self.catalog_columns:
                counts = conn.execute(
                    f"SELECT {col}, COUNT(*) as count FROM catalog GROUP BY {col} ORDER BY count DESC"
                ).fetchall()
                column_stats[col] = dict(counts)
        
        return {
            "total_entries": total_count,
            "column_stats": column_stats,
            "data_root": str(self.data_root),
            "catalog_db": str(self.catalog_db_path)
        }
