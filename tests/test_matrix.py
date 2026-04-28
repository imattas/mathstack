"""Unit tests for matrix module."""

import pytest
from mathcore.core.matrix import Matrix


class TestMatrix:
    """Tests for Matrix class."""
    
    def test_matrix_creation(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.rows == 2
        assert m.cols == 2
    
    def test_matrix_addition(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1 + m2
        assert m3.data == [[6, 8], [10, 12]]
    
    def test_matrix_subtraction(self):
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        m3 = m1 - m2
        assert m3.data == [[4, 4], [4, 4]]
    
    def test_matrix_scalar_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        m2 = m * 2
        assert m2.data == [[2, 4], [6, 8]]
    
    def test_matrix_multiplication(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[2, 0], [1, 2]])
        m3 = m1 * m2
        assert m3.data == [[4, 4], [10, 8]]
    
    def test_matrix_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        mt = m.transpose()
        assert mt.rows == 3
        assert mt.cols == 2
        assert mt.data[0][0] == 1
        assert mt.data[2][1] == 6
    
    def test_matrix_determinant_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        det = m.determinant()
        assert det == -2
    
    def test_matrix_trace(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.trace() == 5
    
    def test_matrix_identity(self):
        m = Matrix.identity(3)
        assert m.data[0][0] == 1
        assert m.data[0][1] == 0
        assert m.data[1][1] == 1
    
    def test_matrix_rank(self):
        m = Matrix([[1, 2], [2, 4]])
        assert m.rank() == 1
    
    def test_matrix_is_symmetric(self):
        m1 = Matrix([[1, 2], [2, 1]])
        assert m1.is_symmetric() == True
        
        m2 = Matrix([[1, 2], [3, 4]])
        assert m2.is_symmetric() == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
