"""
Step-by-Step Work Display Module
Shows detailed work and intermediate steps for all mathematical operations.
"""

from typing import List, Optional, Dict, Any


class StepTracker:
    """Track and display step-by-step mathematical work."""
    
    def __init__(self, title: str = ""):
        """Initialize step tracker.
        
        Args:
            title: Title of the calculation
        """
        self.title = title
        self.steps: List[Dict[str, str]] = []
        self.final_result = None
    
    def add_step(self, description: str, expression: str, result: Optional[str] = None) -> None:
        """Add a step to the calculation.
        
        Args:
            description: What this step does
            expression: The mathematical expression
            result: The result of this step
        """
        step = {
            'description': description,
            'expression': expression,
            'result': result
        }
        self.steps.append(step)
    
    def set_final_result(self, result: Any) -> None:
        """Set the final result."""
        self.final_result = result
    
    def display(self) -> str:
        """Display all steps in a formatted way."""
        output = []
        
        if self.title:
            output.append(f"\n{'='*60}")
            output.append(f"  {self.title}")
            output.append(f"{'='*60}\n")
        
        for i, step in enumerate(self.steps, 1):
            output.append(f"Step {i}: {step['description']}")
            output.append(f"  {step['expression']}")
            if step['result'] is not None:
                output.append(f"  = {step['result']}")
            output.append("")
        
        if self.final_result is not None:
            output.append(f"{'─'*60}")
            output.append(f"Final Result: {self.final_result}")
            output.append(f"{'─'*60}\n")
        
        return "\n".join(output)
    
    def print_steps(self) -> None:
        """Print all steps to console."""
        print(self.display())


class SimplificationTracker(StepTracker):
    """Tracker specifically for algebraic simplification."""
    
    def simplify_arithmetic(self, expression: str) -> str:
        """Simplify arithmetic expression showing work."""
        self.add_step("Original expression", expression)
        
        # Basic arithmetic simplification
        try:
            result = eval(expression)
            self.add_step("Evaluate", f"{expression} = {result}", str(result))
            self.set_final_result(result)
            return str(result)
        except:
            return expression


class CalculationTracker(StepTracker):
    """Track calculations with intermediate results."""
    
    def add_calculation(self, name: str, operation: str, operands: List[Any], result: Any) -> None:
        """Add a calculation step.
        
        Args:
            name: Name of operation (e.g., "Addition", "Multiplication")
            operation: Symbol (e.g., "+", "*")
            operands: List of operands
            result: Result of operation
        """
        expression = f" {operation} ".join(str(o) for o in operands)
        self.add_step(name, expression, str(result))


class EquationSolver:
    """Solve equations showing all steps."""
    
    def __init__(self):
        self.tracker = StepTracker("Equation Solver")
    
    def solve_linear(self, a: float, b: float) -> float:
        """Solve linear equation ax + b = 0.
        
        Returns:
            Solution
        """
        self.tracker.add_step(
            "Linear equation form",
            f"{a}x + {b} = 0"
        )
        
        self.tracker.add_step(
            "Subtract constant from both sides",
            f"{a}x = -{b}"
        )
        
        x = -b / a
        self.tracker.add_step(
            "Divide by coefficient",
            f"x = -{b}/{a}",
            str(x)
        )
        
        self.tracker.set_final_result(f"x = {x}")
        return x
    
    def solve_quadratic(self, a: float, b: float, c: float) -> tuple:
        """Solve quadratic equation ax² + bx + c = 0.
        
        Returns:
            Tuple of solutions
        """
        self.tracker.add_step(
            "Quadratic equation form",
            f"{a}x² + {b}x + {c} = 0"
        )
        
        self.tracker.add_step(
            "Calculate discriminant",
            f"Δ = b² - 4ac = {b}² - 4({a})({c})",
            str(b**2 - 4*a*c)
        )
        
        discriminant = b**2 - 4*a*c
        
        import math
        sqrt_disc = math.sqrt(abs(discriminant))
        
        self.tracker.add_step(
            "Quadratic formula",
            f"x = (-b ± √Δ) / (2a)",
            f"x = (-{b} ± {sqrt_disc}) / {2*a}"
        )
        
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        
        self.tracker.add_step(
            "Solutions",
            f"x₁ = {x1}, x₂ = {x2}"
        )
        
        self.tracker.set_final_result(f"x₁ = {x1}, x₂ = {x2}")
        return (x1, x2)
    
    def display_steps(self) -> None:
        """Display all solving steps."""
        self.tracker.print_steps()


class ArithmeticSteps:
    """Show arithmetic operations step by step."""
    
    @staticmethod
    def add(a: float, b: float, show_steps: bool = True) -> float:
        """Add two numbers showing steps."""
        if show_steps:
            print(f"\nAddition: {a} + {b}")
            print(f"  Step 1: {a}")
            print(f"  Step 2: + {b}")
            print(f"  ─" * 20)
            result = a + b
            print(f"  Result: {result}\n")
        else:
            result = a + b
        return result
    
    @staticmethod
    def multiply(a: float, b: float, show_steps: bool = True) -> float:
        """Multiply two numbers showing steps."""
        if show_steps:
            print(f"\nMultiplication: {a} × {b}")
            print(f"  Step 1: {a}")
            print(f"  Step 2: × {b}")
            print(f"  ─" * 20)
            result = a * b
            print(f"  Result: {result}\n")
        else:
            result = a * b
        return result
    
    @staticmethod
    def divide(a: float, b: float, show_steps: bool = True) -> float:
        """Divide two numbers showing steps."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        if show_steps:
            print(f"\nDivision: {a} ÷ {b}")
            print(f"  Step 1: {a}")
            print(f"  Step 2: ÷ {b}")
            print(f"  ─" * 20)
            result = a / b
            print(f"  Result: {result}\n")
        else:
            result = a / b
        return result
    
    @staticmethod
    def power(base: float, exponent: float, show_steps: bool = True) -> float:
        """Calculate power showing steps."""
        if show_steps:
            print(f"\nExponentiation: {base}^{exponent}")
            print(f"  Base: {base}")
            print(f"  Exponent: {exponent}")
            print(f"  ─" * 20)
            result = base ** exponent
            print(f"  Result: {result}\n")
        else:
            result = base ** exponent
        return result


class ProbabilitySolver:
    """Solve probability problems showing work."""
    
    def __init__(self):
        self.tracker = StepTracker("Probability Calculator")
    
    def binomial_probability(self, n: int, k: int, p: float) -> float:
        """Calculate P(X = k) for binomial distribution."""
        self.tracker.add_step(
            "Binomial probability setup",
            f"n = {n}, k = {k}, p = {p}"
        )
        
        from mathcore.core.arithmetic import binomial_coefficient
        c_nk = binomial_coefficient(n, k)
        
        self.tracker.add_step(
            "Calculate binomial coefficient",
            f"C(n,k) = C({n},{k})",
            str(c_nk)
        )
        
        prob = c_nk * (p ** k) * ((1 - p) ** (n - k))
        
        self.tracker.add_step(
            "Apply binomial formula",
            f"P(X={k}) = C({n},{k}) × {p}^{k} × {1-p}^{n-k}",
            str(prob)
        )
        
        self.tracker.set_final_result(f"P(X={k}) = {prob:.6f}")
        return prob
    
    def display_steps(self) -> None:
        """Display all steps."""
        self.tracker.print_steps()
