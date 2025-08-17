"""
Serializer interface and implementations for data storage system.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import scipy.sparse as sp
import os


class Serializer(ABC):
    """
    Abstract base class for data serializers.
    
    This interface defines the contract for all serializers in the system.
    Serializers handle the actual CRUD operations on data files.
    """
    
    @abstractmethod
    def save(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """
        Save data dictionary to file.
        
        Args:
            file_path: Path where data should be saved
            data_dict: Dictionary containing data to save
        """
        pass
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from file.
        
        Args:
            file_path: Path to file to load
            
        Returns:
            Dictionary containing loaded data
        """
        pass
    
    @abstractmethod
    def delete(self, file_path: Union[str, Path]) -> None:
        """
        Delete file.
        
        Args:
            file_path: Path to file to delete
        """
        pass
    
    @abstractmethod
    def update(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """
        Update existing file with new data.
        
        Args:
            file_path: Path to file to update
            data_dict: Dictionary containing updated data
        """
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension used by this serializer."""
        pass


class NumpyArraySerializer(Serializer):
    """
    Serializer for numpy arrays using .npy files.
    
    This serializer handles:
    - numpy.ndarray (numeric, string, boolean arrays)
    - Does NOT handle scipy sparse matrices (use SparseMatrixSerializer for those)
    
    Features:
    - Preserves exact array dtypes and shapes
    - Uses numpy's native binary format (.npy)
    - No pickle dependency - pure numpy serialization
    - Simple single file per array
    """
    
    @property
    def file_extension(self) -> str:
        """Return the file extension used by this serializer."""
        return ".npy"
    
    def _is_sparse_matrix(self, value: Any) -> bool:
        """
        Check if value is a scipy sparse matrix by checking its type.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is a sparse matrix type
        """
        # Check if it's a scipy sparse matrix using the base class
        return isinstance(value, sp.spmatrix)
    
    def save(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """
        Save data dictionary as .npy files.
        
        For single item dictionaries, saves directly as a .npy file.
        For multiple items, saves as {key}.npy files in a directory.
        
        Args:
            file_path: Base path where data should be saved
            data_dict: Dictionary containing numpy arrays
            
        Raises:
            ValueError: If data contains non-numpy arrays or sparse matrices
        """
        file_path = Path(file_path)
        
        # Validate all data
        for key, value in data_dict.items():
            if isinstance(value, sp.spmatrix):
                raise ValueError(f"Sparse matrices not supported by NumpyArraySerializer. "
                               f"Use SparseMatrixSerializer instead. Key: '{key}'")
            
            # Convert to numpy array if not already
            if not isinstance(value, np.ndarray):
                try:
                    data_dict[key] = np.array(value)
                except Exception as e:
                    raise ValueError(f"Could not convert value to numpy array. Key: '{key}'. Error: {e}")
        
        # Handle single item case (common with AutoSerializer)
        if len(data_dict) == 1:
            key, value = next(iter(data_dict.items()))
            np.save(file_path, value)
        else:
            # Handle multiple items - create directory
            file_path.mkdir(parents=True, exist_ok=True)
            for key, value in data_dict.items():
                array_path = file_path / f"{key}.npy"
                np.save(array_path, value)
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from .npy file or directory containing .npy files.
        
        Args:
            file_path: Path to .npy file or directory containing .npy files
            
        Returns:
            Dictionary containing loaded numpy arrays
            
        Raises:
            FileNotFoundError: If file/directory doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        # Handle single .npy file case
        if file_path.is_file() and file_path.suffix == '.npy':
            array = np.load(file_path, allow_pickle=False)
            return {'data': array}  # Use 'data' as default key for single items
        
        # Handle directory case (multiple .npy files)
        if file_path.is_dir():
            npy_files = list(file_path.glob("*.npy"))
            
            if not npy_files:
                raise ValueError(f"No .npy files found in directory: {file_path}")
            
            # Load all arrays
            result = {}
            for npy_file in npy_files:
                key = npy_file.stem  # Filename without extension
                try:
                    result[key] = np.load(npy_file, allow_pickle=False)
                except Exception as e:
                    raise ValueError(f"Failed to load {npy_file}: {e}")
            
            return result
        
        raise ValueError(f"Expected .npy file or directory, got: {file_path}")
    
    def delete(self, file_path: Union[str, Path]) -> None:
        """
        Delete the .npy file or directory containing .npy files.
        
        Args:
            file_path: Path to .npy file or directory to delete
        """
        file_path = Path(file_path)
        
        if file_path.exists():
            if file_path.is_file():
                # Single .npy file
                file_path.unlink()
            elif file_path.is_dir():
                # Directory with .npy files
                # Remove all .npy files first
                for npy_file in file_path.glob("*.npy"):
                    npy_file.unlink()
                # Remove directory if it's empty
                try:
                    file_path.rmdir()
                except OSError:
                    # Directory not empty, leave it
                    pass
    
    def update(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """
        Update existing .npy file or directory by replacing data.
        
        Args:
            file_path: Path to .npy file or directory to update
            data_dict: Dictionary containing arrays to update
        """
        file_path = Path(file_path)
        
        # Validate data first
        for key, value in data_dict.items():
            if self._is_sparse_matrix(value):
                raise ValueError(f"Sparse matrices not supported by NumpyArraySerializer. "
                               f"Use SparseMatrixSerializer instead. Key: '{key}'")
            
            # Convert to numpy array if not already
            if not isinstance(value, np.ndarray):
                try:
                    data_dict[key] = np.array(value)
                except Exception as e:
                    raise ValueError(f"Cannot convert data '{key}' to numpy array: {e}")
        
        if not file_path.exists():
            # File doesn't exist, create new one
            self.save(file_path, data_dict)
        else:
            # File exists - determine if it's single file or directory case
            if len(data_dict) == 1 and file_path.is_file() and file_path.suffix == '.npy':
                # Single .npy file case
                key, value = next(iter(data_dict.items()))
                np.save(file_path, value)
            else:
                # Directory case - just recreate everything
                self.delete(file_path)
                self.save(file_path, data_dict)


class SparseMatrixSerializer(Serializer):
    """
    Serializer for scipy sparse matrices and numpy arrays using .npz format.
    
    This serializer handles:
    - scipy.sparse matrices (all formats: csr, csc, coo, etc.)
    - numpy.ndarray (numeric and string arrays)
    - Mixed data dictionaries containing both types
    
    Features:
    - Auto-detects sparse vs dense arrays
    - Preserves array dtypes (including strings)
    - Stores sparse matrices as (data, indices, indptr, shape, format) tuples
    - Handles variable dimensions across saves
    - Uses .npz compression for efficiency
    """
    
    @property
    def file_extension(self) -> str:
        """Return the file extension used by this serializer."""
        return ".npz"
    
    def _is_sparse_matrix(self, value: Any) -> bool:
        """
        Check if value is a scipy sparse matrix by checking its type.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is a sparse matrix type
        """
        # Check if it's a scipy sparse matrix using the base class
        return isinstance(value, sp.spmatrix)
    
    def save(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """
        Save mixed data dictionary to .npz file.
        
        Automatically detects and handles:
        - scipy sparse matrices -> stored as structured data
        - numpy arrays -> stored directly
        
        Args:
            file_path: Path where data should be saved
            data_dict: Dictionary containing numpy/scipy data
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process data for saving
        save_dict = {}
        
        for key, value in data_dict.items():
            if self._is_sparse_matrix(value):
                # Handle sparse matrices - convert all to CSR format for consistent storage
                csr_value = value.tocsr()
                save_dict[f"{key}_sparse_data"] = csr_value.data
                save_dict[f"{key}_sparse_indices"] = csr_value.indices
                save_dict[f"{key}_sparse_indptr"] = csr_value.indptr
                save_dict[f"{key}_sparse_shape"] = np.array(csr_value.shape)
                save_dict[f"{key}_sparse_format"] = np.array([value.format], dtype='U10')
                save_dict[f"{key}_is_sparse"] = np.array([True])
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays (both numeric and string)
                save_dict[key] = value
                save_dict[f"{key}_is_sparse"] = np.array([False])
            else:
                # Try to convert to numpy array
                try:
                    save_dict[key] = np.array(value)
                    save_dict[f"{key}_is_sparse"] = np.array([False])
                except Exception as e:
                    raise ValueError(f"Cannot convert data '{key}' to numpy format: {e}")
        
        # Save using numpy's compressed format
        np.savez_compressed(file_path, **save_dict)
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from .npz file and reconstruct original data types.
        
        Args:
            file_path: Path to file to load
            
        Returns:
            Dictionary containing reconstructed data
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the .npz file
        loaded = np.load(file_path, allow_pickle=True)
        
        # Reconstruct original data
        result = {}
        processed_keys = set()
        
        for key in loaded.files:
            if key.endswith('_is_sparse'):
                base_key = key[:-10]  # Remove '_is_sparse' suffix
                
                if base_key in processed_keys:
                    continue
                    
                processed_keys.add(base_key)
                
                is_sparse = loaded[key].item()
                
                if is_sparse:
                    # Reconstruct sparse matrix
                    data = loaded[f"{base_key}_sparse_data"]
                    indices = loaded[f"{base_key}_sparse_indices"]
                    indptr = loaded[f"{base_key}_sparse_indptr"]
                    shape = tuple(loaded[f"{base_key}_sparse_shape"])
                    format_str = loaded[f"{base_key}_sparse_format"].item()
                    
                    # Always reconstruct as CSR first (since that's how we stored it)
                    csr_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
                    
                    # Convert to original format if needed
                    if format_str == 'csr':
                        result[base_key] = csr_matrix
                    elif format_str == 'csc':
                        result[base_key] = csr_matrix.tocsc()
                    elif format_str == 'coo':
                        result[base_key] = csr_matrix.tocoo()
                    else:
                        # For other formats, convert from CSR
                        result[base_key] = csr_matrix.asformat(format_str)
                else:
                    # Regular numpy array
                    result[base_key] = loaded[base_key]
        
        return result
    
    def delete(self, file_path: Union[str, Path]) -> None:
        """
        Delete the .npz file.
        
        Args:
            file_path: Path to file to delete
        """
        file_path = Path(file_path)
        
        if file_path.exists():
            file_path.unlink()
    
    def update(self, file_path: Union[str, Path], data_dict: Dict[str, Any]) -> None:
        """
        Update existing file by merging with new data.
        
        If file doesn't exist, creates a new one.
        
        Args:
            file_path: Path to file to update
            data_dict: Dictionary containing updated data
        """
        file_path = Path(file_path)
        
        if file_path.exists():
            # Load existing data
            existing_data = self.load(file_path)
            # Merge with new data (new data takes precedence)
            existing_data.update(data_dict)
            # Save merged data
            self.save(file_path, existing_data)
        else:
            # File doesn't exist, create new one
            self.save(file_path, data_dict)


class AutoSerializer(Serializer):
    """
    Automatic serializer that chooses between sparse and numpy serializers.
    
    Automatically selects:
    - SparseMatrixSerializer: For scipy sparse matrices
    - NumpyArraySerializer: For numpy arrays
    
    Only supports individual scipy.sparse matrices and numpy arrays.
    """
    
    def __init__(self):
        """Initialize the AutoSerializer."""
        # Initialize available serializers
        self.sparse_serializer = SparseMatrixSerializer()
        self.numpy_serializer = NumpyArraySerializer()
        
        # Track which serializer was used for each file (for loading)
        self._serializer_hints = {}
    
    @property
    def file_extension(self) -> str:
        """Return a generic extension that will be determined by the chosen serializer."""
        return ".auto"
    
    def _choose_serializer(self, data: Any) -> str:
        """
        Choose serializer based on data type.
        
        Args:
            data: Data to analyze (sp.spmatrix or np.ndarray)
            
        Returns:
            'sparse' or 'numpy'
        """
        if isinstance(data, sp.spmatrix):
            return 'sparse'
        elif isinstance(data, np.ndarray):
            return 'numpy'
        else:
            raise ValueError(f"Data must be either scipy sparse matrix or numpy array, got {type(data)}")
    
    def _get_actual_path(self, file_path: Union[str, Path], serializer_type: str) -> Path:
        """
        Generate the actual file path with the correct extension.
        
        Args:
            file_path: Base file path
            serializer_type: 'sparse' or 'numpy'
            
        Returns:
            Actual path with correct extension
        """
        file_path = Path(file_path)
        
        # Remove .auto extension if present
        if file_path.suffix == '.auto':
            file_path = file_path.with_suffix('')
        
        # Add appropriate extension
        if serializer_type == 'sparse':
            return file_path.with_suffix('.npz')
        elif serializer_type == 'numpy':
            return file_path.with_suffix('.npy')
        else:
            raise ValueError(f"Unknown serializer type: {serializer_type}")
    
    def _detect_serializer_from_path(self, file_path: Union[str, Path]) -> tuple[str, Path]:
        """
        Detect which serializer to use based on file path/extension.
        
        Args:
            file_path: File path to analyze
            
        Returns:
            Tuple of (serializer_type, actual_path)
        """
        file_path = Path(file_path)
        
        # Check actual file system
        npz_path = file_path.with_suffix('.npz')
        npy_path = file_path.with_suffix('.npy')
        
        if npz_path.exists():
            return 'sparse', npz_path
        elif npy_path.exists():
            return 'numpy', npy_path
        else:
            # Check stored hint
            hint_key = str(file_path)
            if hint_key in self._serializer_hints:
                serializer_type = self._serializer_hints[hint_key]
                return serializer_type, self._get_actual_path(file_path, serializer_type)
            
            # Default fallback - check both possibilities
            if file_path.suffix == '.npz':
                return 'sparse', file_path
            elif file_path.suffix == '.npy':
                return 'numpy', file_path
            else:
                raise FileNotFoundError(f"Could not find data file for: {file_path}")
    
    def save(self, file_path: Union[str, Path], data: Any) -> None:
        """
        Save data using automatically selected serializer.
        
        Args:
            file_path: Path where data should be saved
            data: scipy sparse matrix or numpy array
        """
        # Choose serializer based on data type
        serializer_type = self._choose_serializer(data)
        
        # Get actual path with correct extension
        actual_path = self._get_actual_path(file_path, serializer_type)
        
        # Store hint for future loading
        self._serializer_hints[str(file_path)] = serializer_type
        
        # Create single-item dictionary for the underlying serializers
        data_dict = {'data': data}
        
        # Delegate to appropriate serializer
        if serializer_type == 'sparse':
            self.sparse_serializer.save(actual_path, data_dict)
        else:  # numpy
            self.numpy_serializer.save(actual_path, data_dict)
    
    def load(self, file_path: Union[str, Path]) -> Any:
        """
        Load data using automatically detected serializer.
        
        Args:
            file_path: Path to file to load
            
        Returns:
            scipy sparse matrix or numpy array
        """
        # Detect which serializer was used
        serializer_type, actual_path = self._detect_serializer_from_path(file_path)
        
        # Delegate to appropriate serializer and extract the single data item
        if serializer_type == 'sparse':
            data_dict = self.sparse_serializer.load(actual_path)
        else:  # numpy
            data_dict = self.numpy_serializer.load(actual_path)
        
        # Return the single data item
        return data_dict['data']
    
    def delete(self, file_path: Union[str, Path]) -> None:
        """
        Delete data using automatically detected serializer.
        
        Args:
            file_path: Path to file to delete
        """
        try:
            # Detect which serializer was used
            serializer_type, actual_path = self._detect_serializer_from_path(file_path)
            
            # Delegate to appropriate serializer
            if serializer_type == 'sparse':
                self.sparse_serializer.delete(actual_path)
            else:  # numpy
                self.numpy_serializer.delete(actual_path)
                
        except FileNotFoundError:
            # File doesn't exist, nothing to delete
            pass
        
        # Clean up hint
        hint_key = str(file_path)
        if hint_key in self._serializer_hints:
            del self._serializer_hints[hint_key]
    
    def update(self, file_path: Union[str, Path], data: Any) -> None:
        """
        Update existing file using the appropriate serializer.
        
        If file doesn't exist, analyze data to choose serializer.
        If file exists, use the same serializer that was used originally.
        
        Args:
            file_path: Path to file to update
            data: scipy sparse matrix or numpy array
        """
        try:
            # Try to detect existing serializer
            serializer_type, actual_path = self._detect_serializer_from_path(file_path)
            
            # Create single-item dictionary for the underlying serializers
            data_dict = {'data': data}
            
            # Delegate to appropriate serializer
            if serializer_type == 'sparse':
                self.sparse_serializer.update(actual_path, data_dict)
            else:  # numpy
                self.numpy_serializer.update(actual_path, data_dict)
                
        except FileNotFoundError:
            # File doesn't exist, create new using analyzed data
            self.save(file_path, data)
    
    def get_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about how the data was stored.
        
        Args:
            file_path: Path to inspect
            
        Returns:
            Dictionary with information about storage method
        """
        try:
            serializer_type, actual_path = self._detect_serializer_from_path(file_path)
            
            info = {
                'serializer_used': serializer_type,
                'actual_path': str(actual_path),
                'exists': actual_path.exists()
            }
            
            if serializer_type == 'sparse':
                info.update({
                    'format': 'npz (compressed)',
                    'description': 'Single file with sparse matrix and numpy array support'
                })
            elif serializer_type == 'numpy':
                info.update({
                    'format': 'npy (single file)',
                    'description': 'Individual .npy file (no pickle)'
                })
                
                # Add file info if it exists
                if actual_path.exists():
                    if actual_path.is_file():
                        info['file_size'] = actual_path.stat().st_size
                    elif actual_path.is_dir():
                        npy_files = list(actual_path.glob('*.npy'))
                        info['num_arrays'] = len(npy_files)
                        info['array_files'] = [f.name for f in npy_files]
            
            return info
            
        except FileNotFoundError:
            return {
                'serializer_used': 'unknown',
                'actual_path': 'not found',
                'exists': False
            }
