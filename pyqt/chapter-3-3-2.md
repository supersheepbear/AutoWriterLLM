# 3.2 Custom Widgets and Painting

# 3.2 Custom Widgets and Painting

In this section, we will explore the creation of custom widgets and how to perform custom painting in Python applications, particularly using a GUI toolkit like Qt (via PyQt or PySide). Custom widgets allow developers to extend the functionality and appearance of standard widgets, while custom painting provides fine-grained control over how a widget is rendered.

## Creating a Custom Widget

Custom widgets are typically created by subclassing an existing widget class and overriding its methods. This allows us to define custom behavior and appearance.

### Basic Custom Widget Example

Let's start with a simple example of a custom widget that draws a circle.

```python
import sys
import logging
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QPoint

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class CircleWidget(QWidget):
    """
    A custom widget that draws a circle in its center.

    Attributes:
        diameter (int): The diameter of the circle.
    """

    def __init__(self, diameter: int = 100, *args, **kwargs):
        """
        Initialize the CircleWidget with a given diameter.

        Args:
            diameter (int): The diameter of the circle. Defaults to 100.
        """
        super().__init__(*args, **kwargs)
        self.diameter = max(1, diameter)  # Ensure diameter is at least 1
        self.setFixedSize(self.diameter, self.diameter)

    def paintEvent(self, event):
        """
        Overrides the paintEvent method to perform custom painting.

        Args:
            event: The paint event object.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw a circle
        painter.setBrush(QColor(255, 165, 0))  # Orange brush
        painter.drawEllipse(0, 0, self.diameter, self.diameter)

        logging.debug("Paint event executed for CircleWidget")

# Application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Handle edge case: Ensure diameter is valid
    try:
        circle_diameter = int(input("Enter circle diameter (default 100): ") or "100")
        if circle_diameter < 1:
            raise ValueError("Diameter must be at least 1")
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
        sys.exit(1)

    # Create and show the widget
    window = CircleWidget(diameter=circle_diameter)
    window.show()

    sys.exit(app.exec_())
```

### Explanation

- **Logging and Error Handling**: We've included logging to track the execution and handle errors, especially around user input for the circle's diameter.
- **Type Hints and Docstrings**: All methods and attributes are type-hinted and documented using Google-style docstrings.
- **Custom Painting**: The `paintEvent` method is overridden to perform custom drawing using `QPainter`.
- **Edge Case Handling**: Input validation ensures that the diameter is at least 1.

## Advanced Custom Painting

Custom painting can be extended to more complex shapes and interactions. Let's modify our widget to support dragging the circle.

```python
from PyQt5.QtCore import QPoint, Qt

class DraggableCircleWidget(CircleWidget):
    """
    A custom widget that allows dragging the circle within the window.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DraggableCircleWidget."""
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.center = QPoint(self.diameter // 2, self.diameter // 2)
        self.dragging = False

    def mousePressEvent(self, event):
        """
        Handle mouse press events to start dragging.

        Args:
            event: The mouse press event object.
        """
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.center = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events to update the circle's position.

        Args:
            event: The mouse move event object.
        """
        if self.dragging:
            self.center = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events to stop dragging.

        Args:
            event: The mouse release event object.
        """
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.update()

    def paintEvent(self, event):
        """
        Overrides the paintEvent method to draw the draggable circle.

        Args:
            event: The paint event object.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the circle at the current center position
        painter.setBrush(QColor(255, 165, 0))
        painter.drawEllipse(self.center - QPoint(self.diameter // 2, self.diameter // 2), self.diameter, self.diameter)

        logging.debug("Paint event executed for DraggableCircleWidget")
```

### Explanation

- **Dragging Logic**: We've added event handlers for mouse press, move, and release to support dragging.
- **Custom Painting Extension**: The `paintEvent` method is updated to draw the circle at the current center position.
- **Edge Case Handling**: We ensure that dragging only occurs when the left mouse button is pressed.

## Best Practices and Common Pitfalls

### Best Practices
- **Use Logging**: Always include logging to track the execution and debug issues.
- **Type Hints and Docstrings**: Use type hints and comprehensive docstrings for better code readability and documentation.
- **Handle Edge Cases**: Validate inputs and handle edge cases to make your application robust.

### Common Pitfalls
- **Not Calling `update()`**: Forgetting to call `update()` in custom painting can result in the widget not redrawing properly.
- **Ignoring Event Handling**: Overlooking event handling methods can lead to unresponsive widgets.
- **Poor Input Validation**: Always validate user inputs to prevent unexpected behavior.

## Practice Exercises

1. **Modify the CircleWidget**: Add functionality to change the circle's color when clicked.
2. **Extend DraggableCircleWidget**: Implement resizing of the circle using the mouse wheel.
3. **Create a Custom Polygon Widget**: Create a widget that draws a regular polygon (e.g., pentagon, hexagon) and supports custom painting and dragging.

## Key Takeaways and Summary

- **Custom Widgets**: Custom widgets allow for the creation of unique and reusable components.
- **Custom Painting**: Overriding `paintEvent` provides control over a widget's appearance.
- **Event Handling**: Implementing event handlers enables interactive features like dragging.
- **Best Practices**: Use logging, type hints, and comprehensive error handling to build robust applications.

By mastering custom widgets and painting, you can significantly enhance the functionality and appearance of your Python applications, providing a richer user experience.
