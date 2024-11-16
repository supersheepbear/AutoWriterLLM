# 2.4 Common Pitfalls and How to Avoid Them

## 2.4 Common Pitfalls and How to Avoid Them

When building PyQt applications, even experienced developers can fall into some common traps that can cause bugs, performance issues, or hard-to-maintain code. In this section, we'll explore these pitfalls and provide strategies to avoid them. We'll continue using Python 3.12, with a focus on best practices, logging, error handling, and type safety.

### Pitfall 1: Not Handling GUI Updates on the Main Thread

PyQt, like most GUI libraries, requires that all UI updates happen on the main thread. Attempting to update the UI from a secondary thread can lead to unpredictable behavior, crashes, or freezes.

#### Example:

```python
import sys
import threading
import time
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QTimer
import logging

logging.basicConfig(level=logging.DEBUG)

class MainWindow(qtw.QWidget):

    def __init__(self):
        super().__init__()
        self.label = qtw.QLabel("Waiting...")
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Simulating a long-running task
        self.worker_thread = threading.Thread(target=self.long_running_task)
        self.worker_thread.start()

    def long_running_task(self):
        """Simulates a long-running task that should update the UI."""
        time.sleep(2)
        # Attempting to update the UI directly from a secondary thread
        self.label.setText("Task Complete!")  # This will cause issues

# Application setup
app = qtw.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

#### Explanation:
- In this example, the `long_running_task` method simulates a time-consuming task using `time.sleep(2)`. After the task completes, the code attempts to update the `QLabel` directly from the secondary thread, which is incorrect and can lead to crashes or unexpected behavior.

#### Solution: Use `QTimer` or `QThread` for Thread-Safe UI Updates

To avoid this, we should use `QTimer` or move the long-running task to a `QThread`, ensuring that UI updates happen on the main thread.

```python
class MainWindow(qtw.QWidget):

    def __init__(self):
        super().__init__()
        self.label = qtw.QLabel("Waiting...")
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Using a timer to simulate a long-running task
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)
        self.timer.start(2000)  # 2 seconds delay

    def on_timeout(self):
        """Callback for timer, safe to update UI here."""
        self.label.setText("Task Complete!")
        logging.debug("UI updated safely from the main thread.")

# Application setup
app = qtw.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

#### Explanation:
- Here, we replaced the secondary thread with `QTimer`. The `on_timeout` method is called after 2 seconds, and it is safe to update the UI from this method since it is executed on the main thread.

#### Best Practice:
- Always use `QTimer` or `QThread` for tasks that need to interact with the UI. Avoid updating the UI directly from secondary threads.

### Pitfall 2: Ignoring Edge Cases and Invalid Inputs

When developing applications, it's crucial to handle edge cases and invalid inputs gracefully. Failing to do so can result in crashes or poor user experience.

#### Example:

```python
def calculate_area(width: float, height: float) -> float:
    """Calculate the area of a rectangle."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be greater than zero.")
    return width * height

class MainWindow(qtw.QWidget):

    def __init__(self):
        super().__init__()
        self.width_input = qtw.QLineEdit(self)
        self.height_input = qtw.QLineEdit(self)
        self.result_label = qtw.QLabel("Result: ", self)

        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.width_input)
        self.layout.addWidget(self.height_input)
        self.calculate_button = qtw.QPushButton("Calculate Area", self)
        self.layout.addWidget(self.calculate_button)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        self.calculate_button.clicked.connect(self.on_calculate)

    def on_calculate(self):
        """Callback for calculate button."""
        try:
            width = float(self.width_input.text())
            height = float(self.height_input.text())
            area = calculate_area(width, height)
            self.result_label.setText(f"Result: {area}")
        except ValueError as e:
            logging.error(f"Invalid input: {e}")
            self.result_label.setText("Result: Invalid input")

# Application setup
app = qtw.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

#### Explanation:
- The `calculate_area` function raises a `ValueError` if the width or height is less than or equal to zero.
- The `on_calculate` method attempts to convert the input text to floats and calculates the area, handling invalid inputs gracefully with error logging and user feedback.

#### Best Practice:
- Always validate user inputs and handle edge cases explicitly to prevent crashes and provide meaningful feedback to users.

### Pitfall 3: Memory Leaks Due to Improper Parent-Child Relationships

PyQt uses a parent-child relationship to manage memory. If you create widgets or other objects without setting their parent, you may end up with memory leaks.

#### Example:

```python
class MainWindow(qtw.QWidget):

    def __init__(self):
        super().__init__()
        # Creating a widget without a parent
        self.floating_widget = qtw.QWidget()
        self.floating_widget.setWindowTitle("This will leak memory")
        self.floating_widget.show()

# Application setup
app = qtw.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

#### Explanation:
- In this example, `floating_widget` does not have a parent, so it will not be properly cleaned up when the main window is closed, leading to a memory leak.

#### Solution:

```python
class MainWindow(qtw.QWidget):

    def __init__(self):
        super().__init__()
        # Assigning self as the parent to manage memory properly
        self.child_widget = qtw.QWidget(self)
        self.child_widget.setWindowTitle("This will not leak memory")
        self.child_widget.show()

# Application setup
app = qtw.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

#### Best Practice:
- Always assign parents to widgets and other PyQt objects to ensure proper memory management.

### Practice Exercises

1. **Threading Exercise**: Modify the `long_running_task` example to use `QThread` instead of `QTimer`.
2. **Input Validation Exercise**: Extend the `calculate_area` function to handle negative inputs by displaying a message box to the user.
3. **Memory Management Exercise**: Create a PyQt application with multiple widgets, ensuring all have appropriate parent-child relationships.

### Key Takeaways and Summary

- **Threading**: Always update the UI from the main thread using `QTimer` or `QThread`.
- **Input Validation**: Validate all user inputs and handle edge cases to prevent crashes and provide a better user experience.
- **Memory Management**: Use parent-child relationships to manage memory effectively and prevent leaks.

By being aware of these common pitfalls and applying the solutions discussed, you can write more robust, maintainable, and efficient PyQt applications.
