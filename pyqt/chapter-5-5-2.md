# 5.2 Debugging Tools and Techniques

# 5.2 Debugging Tools and Techniques

Debugging is an essential skill for developers, allowing them to identify and resolve issues within their code efficiently. Python provides a variety of tools and techniques that can help streamline the debugging process. This section will explore these tools and techniques, emphasizing best practices and practical examples.

## Basic Debugging Tools

### Using `print()` Statements

Though simple, `print()` statements are a powerful initial tool for debugging. They allow developers to inspect the state of variables at various points in the code.

```python
def calculate_average(numbers: list[float]) -> float:
    """
    Calculate the average of a list of numbers.

    Args:
    numbers: A list of floats.

    Returns:
    float: The average of the numbers.
    """
    total = 0.0
    count = 0
    
    for number in numbers:
        total += number
        count += 1
        print(f"Debug: number = {number}, total = {total}, count = {count}")
    
    if count == 0:
        raise ValueError("The list of numbers is empty.")
    
    return total / count

# Example usage
try:
    numbers = [10, 20, 30, 40, 50]
    average = calculate_average(numbers)
    print(f"The average is: {average}")
except ValueError as e:
    print(f"Error: {e}")
```

#### Explanation
- **Design Decision**: The `print()` statements are placed to track the accumulation process, providing insight into each iteration.
- **Best Practice**: Always include error handling, such as raising an exception when dealing with edge cases like an empty list.
- **Common Pitfall**: Over-reliance on `print()` can clutter code; ensure they are temporary and removed after debugging.

### Leveraging `assert` Statements

`assert` statements are useful for checking conditions that should be true during the execution of the program.

```python
def divide(a: float, b: float) -> float:
    """
    Divide two numbers.

    Args:
    a: The dividend.
    b: The divisor.

    Returns:
    float: The result of the division.
    """
    assert b != 0, "Divisor cannot be zero"
    return a / b

# Example usage
try:
    result = divide(10, 0)
    print(result)
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

#### Explanation
- **Design Decision**: The `assert` statement checks for a condition that is critical for the function's correctness.
- **Best Practice**: Use `assert` for defensive programming to catch logical errors early.
- **Common Pitfall**: Asserts are for debugging and testing; they can be disabled in production, so do not rely on them for input validation.

## Advanced Debugging Tools

### Using `pdb` - Python Debugger

The Python Debugger (`pdb`) is an interactive tool that allows setting breakpoints, stepping through code, and inspecting variables.

```python
def find_maximum(numbers: list[float]) -> float:
    """
    Find the maximum number in a list.

    Args:
    numbers: A list of floats.

    Returns:
    float: The maximum number.
    """
    import pdb; pdb.set_trace()  # Breakpoint set here
    max_number = float('-inf')
    for number in numbers:
        if number > max_number:
            max_number = number
    return max_number

# Example usage
numbers = [10, 20, 5, 30, 15]
maximum = find_maximum(numbers)
print(f"The maximum number is: {maximum}")
```

#### Explanation
- **Design Decision**: The `pdb.set_trace()` is used to pause execution and allow interactive debugging.
- **Best Practice**: Use `pdb` for complex issues that require stepping through code and inspecting state.
- **Common Pitfall**: Remember to remove or comment out `pdb` calls after debugging to avoid interrupting program flow.

### Logging for Debugging

Logging provides a more structured and persistent approach to debugging.

```python
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def complex_calculation(a: float, b: float) -> float:
    """
    Perform a complex calculation.

    Args:
    a: First operand.
    b: Second operand.

    Returns:
    float: Result of the calculation.
    """
    logging.debug(f"Starting calculation with a={a}, b={b}")
    result = a ** 2 + b ** 2
    logging.debug(f"Intermediate result: {result}")
    result += result * 0.1  # Some complex logic
    logging.debug(f"Final result: {result}")
    return result

# Example usage
result = complex_calculation(3, 4)
print(f"Calculation result: {result}")
```

#### Explanation
- **Design Decision**: Logging is configured to capture debug-level messages, providing detailed insights into the function's operation.
- **Best Practice**: Use different logging levels to distinguish between informational and debugging messages.
- **Common Pitfall**: Be mindful of logging too much information, which can impact performance and clutter logs.

## Best Practices and Common Pitfalls

### Best Practices
- **Isolate the Problem**: Narrow down the issue to a minimal, reproducible example.
- **Use Version Control**: Leverage Git or other version control systems to experiment safely and revert changes if necessary.
- **Reproduce the Issue**: Ensure you can reproduce the bug consistently before attempting to fix it.

### Common Pitfalls
- **Over-reliance on Print Statements**: While useful, avoid leaving them in production code.
- **Ignoring Edge Cases**: Always consider edge cases and invalid inputs during testing and debugging.
- **Not Reading Error Messages Carefully**: Error messages often contain valuable information pointing to the root cause.

## Practice Exercises

1. **Exercise 1**: Write a function to reverse a list of strings and use `print()` statements to debug the process.
2. **Exercise 2**: Extend the `divide()` function to handle division by floating-point zero and use assertions to validate inputs.
3. **Exercise 3**: Implement a recursive function to compute the factorial of a number and use `pdb` to trace its execution.
4. **Exercise 4**: Enhance the logging example to include timestamps and log rotation for large log files.

## Key Takeaways and Summary

- **Print Statements**: Useful for quick debugging but should be temporary.
- **Assertions**: Help catch logical errors early in development.
- **pdb**: A powerful interactive debugging tool for stepping through code.
- **Logging**: Provides structured and persistent debugging information.
- **Best Practices**: Isolate problems, use version control, reproduce issues, and read error messages carefully.
- **Common Pitfalls**: Avoid over-relying on print statements, ignoring edge cases, and misreading error messages.

By mastering these debugging tools and techniques, you can effectively troubleshoot and resolve issues in your Python applications, enhancing both the quality and reliability of your code.
