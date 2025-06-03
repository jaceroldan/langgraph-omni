import math
import numexpr

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
        Calculate expression using Python's numexpr library.

        Expression should be a single line mathematical expression that solves
        the problem.

        Examples:
            "37593 * 67" for "37593 times 67"
            "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    result = str(numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict=local_dict
        ))
    return f"The calculated result is {result}"
