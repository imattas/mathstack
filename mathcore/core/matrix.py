"""
Matrix operations and linear algebra module.
Supports matrix creation, manipulation, and advanced operations.
"""

import math
from typing import List, Tuple, Optional, Union
from copy import deepcopy


class Matrix:
    """Represents and manipulates matrices."""
    
    def __init__(self, data: List[List[float]]):
        """Initialize matrix from list of lists.
        
        Args:
            data: 2D list representing matrix
        """
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length")
        
        self.data = deepcopy(data)
        self.rows = len(data)
        self.cols = len(data[0])
    
    def __repr__(self):
        result = []
        for row in self.data:
            result.append("[" + ", ".join(f"{x:.2f}" for x in row) + "]")
        return "Matrix([\n  " + "\n  ".join(result) + "\n])"
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions")
        
        result = []
        for i in range(self.rows):
            row = [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            result.append(row)
        return Matrix(result)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions")
        
        result = []
        for i in range(self.rows):
            row = [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            result.append(row)
        return Matrix(result)
    
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        """Multiply matrix by scalar or another matrix."""
        if isinstance(other, (int, float)):
            result = []
            for row in self.data:
                result.append([x * other for x in row])
            return Matrix(result)
        
        # Matrix multiplication
        if self.cols != other.rows:
            raise ValueError("Number of columns in first matrix must equal rows in second")
        
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                val = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                row.append(val)
            result.append(row)
        return Matrix(result)
    
    def transpose(self) -> 'Matrix':
        """Return transpose of matrix."""
        result = []
        for j in range(self.cols):
            row = [self.data[i][j] for i in range(self.rows)]
            result.append(row)
        return Matrix(result)
    
    def determinant(self) -> float:
        """Calculate determinant (only for square matrices)."""
        if self.rows != self.cols:
            raise ValueError("Determinant only defined for square matrices")
        
        if self.rows == 1:
            return self.data[0][0]
        
        if self.rows == 2:
            return (self.data[0][0] * self.data[1][1] - 
                    self.data[0][1] * self.data[1][0])
        
        # Use cofactor expansion for larger matrices
        det = 0
        for j in range(self.cols):
            minor = self._get_minor(0, j)
            cofactor = ((-1) ** j) * minor.determinant()
            det += self.data[0][j] * cofactor
        
        return det
    
    def _get_minor(self, row: int, col: int) -> 'Matrix':
        """Get minor matrix by removing row and col."""
        result = []
        for i in range(self.rows):
            if i == row:
                continue
            new_row = [self.data[i][j] for j in range(self.cols) if j != col]
            result.append(new_row)
        return Matrix(result)
    
    def inverse(self) -> 'Matrix':
        """Calculate inverse of matrix (for square matrices)."""
        if self.rows != self.cols:
            raise ValueError("Inverse only defined for square matrices")
        
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (non-invertible)")
        
        if self.rows == 2:
            result = [
                [self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det, self.data[0][0] / det]
            ]
            return Matrix(result)
        
        # Use adjugate method for larger matrices
        adj = self._adjugate()
        return adj * (1 / det)
    
    def _adjugate(self) -> 'Matrix':
        """Calculate adjugate (adjoint) matrix."""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                minor = self._get_minor(j, i)  # Note: transposed indices
                cofactor = ((-1) ** (i + j)) * minor.determinant()
                row.append(cofactor)
            result.append(row)
        return Matrix(result)
    
    def trace(self) -> float:
        """Calculate trace (sum of diagonal elements)."""
        if self.rows != self.cols:
            raise ValueError("Trace only defined for square matrices")
        return sum(self.data[i][i] for i in range(self.rows))
    
    def is_symmetric(self) -> bool:
        """Check if matrix is symmetric."""
        if self.rows != self.cols:
            return False
        
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - self.data[j][i]) > 1e-10:
                    return False
        return True
    
    def is_orthogonal(self) -> bool:
        """Check if matrix is orthogonal (A * A^T = I)."""
        if self.rows != self.cols:
            return False
        
        product = self * self.transpose()
        identity = Matrix([[1.0 if i == j else 0.0 for j in range(self.rows)] 
                          for i in range(self.rows)])
        
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(product.data[i][j] - identity.data[i][j]) > 1e-10:
                    return False
        return True
    
    def rank(self) -> int:
        """Calculate rank of matrix using Gaussian elimination."""
        matrix = deepcopy(self)
        rank = 0
        
        for col in range(matrix.cols):
            # Find pivot
            pivot_row = -1
            for row in range(rank, matrix.rows):
                if abs(matrix.data[row][col]) > 1e-10:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap rows
            matrix.data[rank], matrix.data[pivot_row] = (
                matrix.data[pivot_row], matrix.data[rank]
            )
            
            # Eliminate column
            for row in range(matrix.rows):
                if row != rank and abs(matrix.data[row][col]) > 1e-10:
                    factor = matrix.data[row][col] / matrix.data[rank][col]
                    for j in range(matrix.cols):
                        matrix.data[row][j] -= factor * matrix.data[rank][j]
            
            rank += 1
        
        return rank
    
    @staticmethod
    def identity(n: int) -> 'Matrix':
        """Create n×n identity matrix."""
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return Matrix(data)
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create zero matrix."""
        return Matrix([[0.0 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def ones(rows: int, cols: int) -> 'Matrix':
        """Create matrix filled with ones."""
        return Matrix([[1.0 for _ in range(cols)] for _ in range(rows)])
    
    def solve_linear_system(self, b: 'Matrix') -> 'Matrix':
        """Solve linear system Ax = b using Gaussian elimination.
        
        Args:
            b: Right-hand side matrix/vector
            
        Returns:
            Solution vector x
        """
        if self.rows != self.cols:
            raise ValueError("Coefficient matrix must be square")
        if self.rows != b.rows:
            raise ValueError("Dimension mismatch")
        
        # Create augmented matrix [A|b]
        augmented = []
        for i in range(self.rows):
            row = self.data[i] + b.data[i]
            augmented.append(row)
        
        n = self.rows
        
        # Forward elimination
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                    max_row = row
            
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
            
            # Eliminate below
            for row in range(col + 1, n):
                if abs(augmented[col][col]) > 1e-10:
                    factor = augmented[row][col] / augmented[col][col]
                    for j in range(col, len(augmented[row])):
                        augmented[row][j] -= factor * augmented[col][j]
        
        # Back substitution
        solution = []
        for i in range(n - 1, -1, -1):
            row_solution = []
            for k in range(b.cols):
                val = augmented[i][n + k]
                for j in range(i + 1, n):
                    val -= augmented[i][j] * solution[n - 1 - j][k]
                if abs(augmented[i][i]) > 1e-10:
                    val /= augmented[i][i]
                row_solution.append(val)
            solution.insert(0, row_solution)
        
        return Matrix(solution)
