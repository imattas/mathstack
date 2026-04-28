"""
Differential Equations Solver Module
ODE solvers: Euler, RK4, stiff equations, systems of equations.
"""

import math
from typing import Callable, List, Tuple


class ODESolver:
    """Ordinary Differential Equation solvers."""
    
    @staticmethod
    def euler_method(f: Callable[[float, float], float], y0: float, 
                    a: float, b: float, n: int) -> Tuple[List[float], List[float]]:
        """Solve dy/dt = f(t, y) using Euler's method.
        
        Args:
            f: Derivative function f(t, y)
            y0: Initial condition y(a)
            a, b: Interval [a, b]
            n: Number of steps
            
        Returns:
            Tuple of (t_values, y_values)
        """
        h = (b - a) / n
        t_values = [a + i * h for i in range(n + 1)]
        y_values = [y0]
        
        for i in range(n):
            t = t_values[i]
            y = y_values[i]
            y_new = y + h * f(t, y)
            y_values.append(y_new)
        
        return (t_values, y_values)
    
    @staticmethod
    def rk2_method(f: Callable[[float, float], float], y0: float,
                  a: float, b: float, n: int) -> Tuple[List[float], List[float]]:
        """Solve using Runge-Kutta 2nd order (midpoint method)."""
        h = (b - a) / n
        t_values = [a + i * h for i in range(n + 1)]
        y_values = [y0]
        
        for i in range(n):
            t = t_values[i]
            y = y_values[i]
            
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h * k1 / 2)
            
            y_new = y + h * k2
            y_values.append(y_new)
        
        return (t_values, y_values)
    
    @staticmethod
    def rk4_method(f: Callable[[float, float], float], y0: float,
                  a: float, b: float, n: int) -> Tuple[List[float], List[float]]:
        """Solve using Runge-Kutta 4th order (most accurate)."""
        h = (b - a) / n
        t_values = [a + i * h for i in range(n + 1)]
        y_values = [y0]
        
        for i in range(n):
            t = t_values[i]
            y = y_values[i]
            
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h * k1 / 2)
            k3 = f(t + h / 2, y + h * k2 / 2)
            k4 = f(t + h, y + h * k3)
            
            y_new = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_values.append(y_new)
        
        return (t_values, y_values)
    
    @staticmethod
    def solve_system_rk4(f_system: Callable[[float, List[float]], List[float]],
                        y0: List[float], a: float, b: float, n: int) -> Tuple[List[float], List[List[float]]]:
        """Solve system of ODEs dy/dt = f(t, y).
        
        Args:
            f_system: Function returning list of derivatives
            y0: Initial conditions
            a, b: Interval
            n: Number of steps
            
        Returns:
            Tuple of (t_values, y_values_matrix)
        """
        h = (b - a) / n
        t_values = [a + i * h for i in range(n + 1)]
        y_values = [y0[:]]
        
        for i in range(n):
            t = t_values[i]
            y = y_values[i]
            
            k1 = f_system(t, y)
            
            y_half = [y[j] + h * k1[j] / 2 for j in range(len(y))]
            k2 = f_system(t + h / 2, y_half)
            
            y_half = [y[j] + h * k2[j] / 2 for j in range(len(y))]
            k3 = f_system(t + h / 2, y_half)
            
            y_full = [y[j] + h * k3[j] for j in range(len(y))]
            k4 = f_system(t + h, y_full)
            
            y_new = [y[j] + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                    for j in range(len(y))]
            y_values.append(y_new)
        
        return (t_values, y_values)


class PartialDifferentialEquations:
    """Simple PDE solvers."""
    
    @staticmethod
    def heat_equation_1d(initial_conditions: List[float], 
                        time_steps: int, space_steps: int,
                        dt: float = 0.001, dx: float = 0.1) -> Tuple[List[List[float]]]:
        """Solve 1D heat equation using finite differences.
        
        Args:
            initial_conditions: Initial temperature distribution
            time_steps: Number of time steps
            space_steps: Number of space steps
            dt: Time step size
            dx: Space step size
            
        Returns:
            Solution matrix [time, space]
        """
        # Stability condition: dt <= dx^2 / (4 * alpha)
        # For alpha = 1, dt <= dx^2 / 4
        r = dt / (dx ** 2)
        
        if r > 0.25:
            raise ValueError("Stability condition violated: dt <= dx²/4")
        
        # Initialize solution
        u = [initial_conditions[:]]
        
        for t in range(1, time_steps):
            u_new = [0] * space_steps
            
            # Boundary conditions (homogeneous Dirichlet)
            u_new[0] = 0
            u_new[-1] = 0
            
            # Interior points
            for i in range(1, space_steps - 1):
                u_new[i] = (r * u[t-1][i+1] + (1 - 2*r) * u[t-1][i] + r * u[t-1][i-1])
            
            u.append(u_new)
        
        return u
    
    @staticmethod
    def wave_equation_1d(initial_displacement: List[float],
                        initial_velocity: List[float],
                        time_steps: int, space_steps: int,
                        dt: float = 0.001, dx: float = 0.1) -> Tuple[List[List[float]]]:
        """Solve 1D wave equation using finite differences."""
        # Speed of wave
        c = 1.0
        r = (c * dt / dx) ** 2
        
        if r > 1:
            raise ValueError("Stability condition violated: c*dt <= dx")
        
        # Initialize
        u = [initial_displacement[:]]
        
        # First time step (using initial velocity)
        u_1 = []
        for i in range(space_steps):
            if i == 0 or i == space_steps - 1:
                u_1.append(0)  # Boundary
            else:
                u_1.append(
                    u[0][i] + dt * initial_velocity[i] +
                    (r / 2) * (u[0][i+1] - 2 * u[0][i] + u[0][i-1])
                )
        u.append(u_1)
        
        # Subsequent time steps
        for t in range(2, time_steps):
            u_new = []
            for i in range(space_steps):
                if i == 0 or i == space_steps - 1:
                    u_new.append(0)  # Boundary
                else:
                    u_new.append(
                        2 * u[t-1][i] - u[t-2][i] +
                        r * (u[t-1][i+1] - 2 * u[t-1][i] + u[t-1][i-1])
                    )
            u.append(u_new)
        
        return u


class StiffODESolver:
    """Solvers for stiff differential equations."""
    
    @staticmethod
    def backward_euler(f: Callable[[float, float], float], 
                      y0: float, a: float, b: float, n: int,
                      max_newton_iterations: int = 10) -> Tuple[List[float], List[float]]:
        """Solve stiff ODE using Backward Euler method."""
        h = (b - a) / n
        t_values = [a + i * h for i in range(n + 1)]
        y_values = [y0]
        
        for i in range(n):
            t = t_values[i]
            y_curr = y_values[i]
            
            # Solve y_new = y_curr + h * f(t_new, y_new) using Newton's method
            y_new = y_curr  # Initial guess
            
            for _ in range(max_newton_iterations):
                # Numerical derivative for Newton's method
                h_deriv = 1e-8
                fy = f(t + h, y_new)
                fy_deriv = (f(t + h, y_new + h_deriv) - fy) / h_deriv
                
                # Newton iteration
                residual = y_new - y_curr - h * fy
                y_new_update = y_new - residual / (1 - h * fy_deriv)
                
                if abs(y_new_update - y_new) < 1e-10:
                    break
                
                y_new = y_new_update
            
            y_values.append(y_new)
        
        return (t_values, y_values)
