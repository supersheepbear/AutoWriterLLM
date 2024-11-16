# 6.2 Final Assessment

# 6.2 Final Assessment

The final assessment of the capstone project is designed to evaluate your ability to integrate the concepts you've learned throughout this tutorial. This section will guide you through a comprehensive evaluation of your project, focusing on code quality, functionality, performance, and documentation. We'll provide examples using Python 3.12, ensuring that you follow best practices, handle errors effectively, and optimize performance without sacrificing clarity.

## 6.2.1 Code Quality and Best Practices

### Comprehensive Logging and Error Handling

Effective logging and error handling are crucial for maintaining and troubleshooting your application. Let's start with a basic example that demonstrates these practices.

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def divide(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        The result of the division

    Raises:
        ValueError: If the divisor is zero

    Examples:
        >>> divide(10, 2)
        5.0
        >>> divide(10, 0)
        Traceback (most recent call last):
        ...
        ValueError: Divisor cannot be zero
    """
    try:
        if b == 0:
            logger.error("Divisor cannot be zero")
            raise ValueError("Divisor cannot be zero")
        return a / b
    except Exception as e:
        logger.exception(e)
        raise

# Example usage
try:
    result = divide(10, 0)
except ValueError:
    logger.info("Handled division by zero")
```

### Design Decisions and Trade-offs

- **Logging**: We use Python's built-in `logging` module to capture errors and important events. This approach provides flexibility in managing log outputs and levels.
- **Error Handling**: By raising and catching specific exceptions, we ensure that errors are meaningful and actionable.
- **Best Practices**: Always log exceptions and handle them gracefully to prevent application crashes.

### Edge Cases and Unusual Scenarios

Handling edge cases is vital for robust software. Below is an example that showcases handling invalid inputs.

```python
def safe_sqrt(x: float) -> float:
    """Calculate the square root of a number.

    Args:
        x: Number to calculate the square root for

    Returns:
        The square root of the number

    Raises:
        ValueError: If the number is negative

    Examples:
        >>> safe_sqrt(4)
        2.0
        >>> safe_sqrt(-1)
        Traceback (most recent call last):
        ...
        ValueError: Cannot calculate square root of a negative number
    """
    if x < 0:
        logger.error("Cannot calculate square root of a negative number")
        raise ValueError("Cannot calculate square root of a negative number")
    return x ** 0.5

# Example usage
try:
    result = safe_sqrt(-1)
except ValueError as e:
    logger.info(f"Handled invalid input: {e}")
```

### Performance Optimization

Optimizing performance while maintaining code clarity can be challenging. Let's look at a function that calculates the Fibonacci sequence efficiently using memoization.

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.

    Args:
        n: The index of the Fibonacci number

    Returns:
        The nth Fibonacci number

    Examples:
        >>> fibonacci(10)
        55
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
result = fibonacci(10)
logger.info(f"10th Fibonacci number: {result}")
```

### Design Decisions and Trade-offs

- **Memoization**: Using `lru_cache` significantly improves performance for recursive functions by caching results of previous computations.
- **Clarity**: The recursive function definition remains straightforward and readable.

## 6.2.2 Documentation and Type Hints

### Google-style Docstrings and Sphinx Documentation

Documenting your code effectively is as important as writing it. Below is an example that demonstrates Google-style docstrings and type hints.

```python
def add_numbers(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First operand
        b: Second operand

    Returns:
        The sum of the two numbers

    Examples:
        >>> add_numbers(3, 4)
        7
    """
    return a + b

# Example usage
result = add_numbers(3, 4)
logger.info(f"Sum of 3 and 4: {result}")
```

### Best Practices

- **Type Hints**: Ensure clarity and catch type-related errors early.
- **Docstrings**: Provide examples and explanations for functions, making it easier for others (and your future self) to understand and use your code.

## Practice Exercises

1. **Logging and Error Handling**: Write a function that reads a file and logs an error if the file does not exist. Include type hints and docstrings.
2. **Performance Optimization**: Implement a sorting function (e.g., bubble sort) and optimize it using a known technique (e.g., early termination).
3. **Documentation**: Document a complex function of your choice using Google-style docstrings and generate Sphinx documentation.

## Key Takeaways and Summary

- **Logging and Error Handling**: Always implement robust logging and handle exceptions gracefully.
- **Edge Cases**: Anticipate and handle edge cases to ensure your application behaves correctly under unusual scenarios.
- **Performance Optimization**: Use techniques like memoization to improve performance without sacrificing readability.
- **Documentation**: Comprehensive documentation using type hints and docstrings is essential for maintainable code.

By integrating these practices into your capstone project, you ensure that your code is not only functional but also robust, maintainable, and understandable. This concludes the final assessment section, and you are now ready to apply these concepts in your capstone project.
