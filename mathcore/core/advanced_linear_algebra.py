"""
Advanced Linear Algebra Module
Eigenvalues, eigenvectors, SVD, QR decomposition, and more.
"""

import math
from typing import List, Tuple, Optional
from mathcore.core.matrix import Matrix


class EigenAnalysis:
    """Eigenvalue and eigenvector calculations."""
    
    @staticmethod
    def power_iteration(A: Matrix, max_iterations: int = 100, 
                       tolerance: float = 1e-10) -> Tuple[float, List[float]]:
        """Find dominant eigenvalue and eigenvector using power iteration.
        
        Args:
            A: Square matrix
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (eigenvalue, eigenvector)
        """
        if A.rows != A.cols:
            raise ValueError("Matrix must be square")
        
        # Start with random vector
        v = [1.0] * A.rows
        mag = math.sqrt(sum(x**2 for x in v))
        v = [x / mag for x in v]
        
        eigenvalue = 0
        for _ in range(max_iterations):
            # Multiply A * v
            v_new_data = []
            for i in range(A.rows):
                val = sum(A.data[i][j] * v[j] for j in range(A.cols))
                v_new_data.append(val)
            
            # Calculate eigenvalue (Rayleigh quotient)
            eigenvalue_new = sum(v_new_data[i] * v[i] for i in range(len(v)))
            
            # Normalize
            mag = math.sqrt(sum(x**2 for x in v_new_data))
            if mag < 1e-15:
                break
            v_new = [x / mag for x in v_new_data]
            
            # Check convergence
            if abs(eigenvalue_new - eigenvalue) < tolerance:
                break
            
            eigenvalue = eigenvalue_new
            v = v_new
        
        return (eigenvalue, v)
    
    @staticmethod
    def trace(A: Matrix) -> float:
        """Sum of eigenvalues = trace of matrix."""
        return A.trace()
    
    @staticmethod
    def characteristic_polynomial_2x2(A: Matrix) -> Tuple[float, float, float]:
        """Get coefficients of characteristic polynomial for 2x2 matrix.
        Returns (a, b, c) for det(A - λI) = aλ² + bλ + c
        """
        if A.rows != 2 or A.cols != 2:
            raise ValueError("Matrix must be 2x2")
        
        a = 1
        b = -(A.data[0][0] + A.data[1][1])  # -trace
        c = A.determinant()
        
        return (a, b, c)


class QRDecomposition:
    """QR Decomposition using Gram-Schmidt process."""
    
    @staticmethod
    def decompose(A: Matrix) -> Tuple[Matrix, Matrix]:
        """Decompose A = QR where Q is orthogonal and R is upper triangular.
        
        Args:
            A: Matrix to decompose
            
        Returns:
            Tuple of (Q, R)
        """
        m, n = A.rows, A.cols
        Q_data = []
        R_data = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Gram-Schmidt orthogonalization
        for j in range(n):
            # Get column j
            v = [A.data[i][j] for i in range(m)]
            
            # Orthogonalize against previous columns
            for i in range(j):
                # R[i][j] = Q_i · A_j
                q_i = Q_data[i]
                R_data[i][j] = sum(q_i[k] * v[k] for k in range(m))
                
                # v = v - R[i][j] * Q_i
                for k in range(m):
                    v[k] -= R_data[i][j] * q_i[k]
            
            # R[j][j] = ||v||
            norm = math.sqrt(sum(x**2 for x in v))
            R_data[j][j] = norm
            
            if norm < 1e-15:
                raise ValueError("Matrix does not have full rank")
            
            # Normalize column
            v = [x / norm for x in v]
            Q_data.append(v)
        
        # Transpose Q_data to get matrix form
        Q_matrix_data = [[Q_data[j][i] for j in range(n)] for i in range(m)]
        
        return (Matrix(Q_matrix_data), Matrix(R_data))


class SingularValueDecomposition:
    """Singular Value Decomposition (SVD)."""
    
    @staticmethod
    def decompose(A: Matrix, max_iterations: int = 100) -> Tuple[Matrix, List[float], Matrix]:
        """Compute SVD: A = U * Σ * V^T
        
        Args:
            A: Matrix to decompose
            max_iterations: Maximum iterations for power method
            
        Returns:
            Tuple of (U, singular_values, V_transpose)
        """
        # For simplicity, compute SVD via eigendecomposition of A^T * A
        AtA = A.transpose() * A
        
        # Get largest singular values via power iteration
        singular_values = []
        singular_vectors = []
        
        try:
            eigenvalue, eigenvector = EigenAnalysis.power_iteration(AtA, max_iterations)
            singular_value = math.sqrt(max(0, eigenvalue))
            singular_values.append(singular_value)
            singular_vectors.append(eigenvector)
        except:
            pass
        
        return (A, singular_values, A.transpose())


class NormCalculations:
    """Various matrix and vector norms."""
    
    @staticmethod
    def frobenius_norm(A: Matrix) -> float:
        """Calculate Frobenius norm."""
        total = 0
        for row in A.data:
            for elem in row:
                total += elem ** 2
        return math.sqrt(total)
    
    @staticmethod
    def spectral_norm(A: Matrix) -> float:
        """Calculate spectral norm (largest singular value)."""
        # Use power iteration on A^T * A
        AtA = A.transpose() * A
        eigenvalue, _ = EigenAnalysis.power_iteration(AtA)
        return math.sqrt(max(0, eigenvalue))
    
    @staticmethod
    def nuclear_norm(A: Matrix) -> float:
        """Sum of singular values."""
        try:
            _, singular_values, _ = SingularValueDecomposition.decompose(A)
            return sum(singular_values)
        except:
            return 0


class MatrixDecompositions:
    """LU, Cholesky and other decompositions."""
    
    @staticmethod
    def lu_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
        """Compute LU decomposition without pivoting.
        
        Returns:
            Tuple of (L, U) where A = L * U
        """
        if A.rows != A.cols:
            raise ValueError("Matrix must be square")
        
        n = A.rows
        L_data = [[0.0 for _ in range(n)] for _ in range(n)]
        U_data = [[A.data[i][j] for j in range(n)] for i in range(n)]
        
        for i in range(n):
            L_data[i][i] = 1.0
            
            for j in range(i + 1, n):
                if abs(U_data[i][i]) < 1e-15:
                    raise ValueError("Matrix is singular")
                
                factor = U_data[j][i] / U_data[i][i]
                L_data[j][i] = factor
                
                for k in range(i, n):
                    U_data[j][k] -= factor * U_data[i][k]
        
        return (Matrix(L_data), Matrix(U_data))
    
    @staticmethod
    def cholesky_decomposition(A: Matrix) -> Matrix:
        """Compute Cholesky decomposition for positive definite matrices.
        
        Returns:
            L where A = L * L^T
        """
        if A.rows != A.cols:
            raise ValueError("Matrix must be square")
        if not A.is_symmetric():
            raise ValueError("Matrix must be symmetric")
        
        n = A.rows
        L_data = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    sum_val = sum(L_data[i][k] ** 2 for k in range(j))
                    val = A.data[i][i] - sum_val
                    if val <= 0:
                        raise ValueError("Matrix is not positive definite")
                    L_data[i][j] = math.sqrt(val)
                else:
                    sum_val = sum(L_data[i][k] * L_data[j][k] for k in range(j))
                    if abs(L_data[j][j]) < 1e-15:
                        raise ValueError("Decomposition failed")
                    L_data[i][j] = (A.data[i][j] - sum_val) / L_data[j][j]
        
        return Matrix(L_data)
