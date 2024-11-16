# 6.1 Capstone Project: Building a Complex Application

# 6.1 Capstone Project: Building a Complex Application

In this section, we will guide you through building a complex application as part of the capstone project. The aim is to integrate the knowledge you've acquired throughout the previous chapters and apply it to a real-world scenario. We will focus on developing a robust, maintainable, and efficient application while adhering to best practices in Python development.

## Project Overview

The capstone project involves creating a RESTful API for a fictional library system. The API will allow clients to perform CRUD (Create, Read, Update, Delete) operations on books and users. We'll use Python 3.12, the `FastAPI` framework, and an in-memory database for simplicity.

### Requirements

1. **API Endpoints**:
    - `GET /books`: Retrieve a list of all books.
    - `POST /books`: Add a new book.
    - `PUT /books/{book_id}`: Update an existing book.
    - `DELETE /books/{book_id}`: Delete a book.
    - `GET /users`: Retrieve a list of all users.
    - `POST /users`: Add a new user.
    - `PUT /users/{user_id}`: Update an existing user.
    - `DELETE /users/{user_id}`: Delete a user.

2. **Comprehensive Logging and Error Handling**:
    - Log all requests and responses.
    - Handle and log exceptions gracefully.

3. **Edge Cases and Invalid Inputs**:
    - Handle invalid book and user IDs.
    - Validate input data for POST and PUT requests.

4. **Performance Optimization**:
    - Optimize the in-memory data structure for quick data retrieval.

5. **Type Hints**:
    - Use type hints for all variables and functions.

6. **Documentation**:
    - Provide Google-style docstrings for Sphinx documentation.

## Project Setup

First, ensure you have Python 3.12 installed. Then, create a virtual environment and install the necessary packages:

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn
```

## Code Implementation

### Data Models

We'll start by defining data models for books and users using Python's `dataclass` and type hints.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Union
from uuid import UUID, uuid4

@dataclass
class Book:
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    author: str = ""
    isbn: str = ""

@dataclass
class User:
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    email: str = ""

# In-memory database
database: Dict[str, List[Union[Book, User]]] = {
    "books": [],
    "users": [],
}
```

### Logging and Error Handling

We'll configure logging to capture both requests and errors.

```python
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(HTTPException)
async def custom_exception_handler(request, exc):
    logger.error(f"Error: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.middleware("http")
async def log_request(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Outgoing response: {response.status_code}")
    return response
```

### API Endpoints

#### Books

```python
from fastapi import HTTPException
from uuid import UUID

@app.get("/books", tags=["books"])
def get_books() -> List[Book]:
    """
    Retrieve a list of all books.

    Returns:
        List[Book]: A list of Book objects.
    """
    return database["books"]

@app.post("/books", tags=["books"])
def add_book(book: Book) -> dict:
    """
    Add a new book to the library.

    Args:
        book (Book): The Book object to be added.

    Returns:
        dict: A message confirming the addition of the book.
    """
    database["books"].append(book)
    return {"message": "Book added successfully"}

@app.put("/books/{book_id}", tags=["books"])
def update_book(book_id: UUID, book: Book) -> dict:
    """
    Update an existing book in the library.

    Args:
        book_id (UUID): The ID of the book to be updated.
        book (Book): The updated Book object.

    Returns:
        dict: A message confirming the update of the book.
    """
    for idx, b in enumerate(database["books"]):
        if b.id == book_id:
            database["books"][idx] = book
            return {"message": "Book updated successfully"}
    raise HTTPException(status_code=404, detail="Book not found")

@app.delete("/books/{book_id}", tags=["books"])
def delete_book(book_id: UUID) -> dict:
    """
    Delete a book from the library.

    Args:
        book_id (UUID): The ID of the book to be deleted.

    Returns:
        dict: A message confirming the deletion of the book.
    """
    for idx, b in enumerate(database["books"]):
        if b.id == book_id:
            del database["books"][idx]
            return {"message": "Book deleted successfully"}
    raise HTTPException(status_code=404, detail="Book not found")
```

#### Users

The implementation for users follows the same pattern as books. Here's an example for adding and retrieving users:

```python
@app.get("/users", tags=["users"])
def get_users() -> List[User]:
    """
    Retrieve a list of all users.

    Returns:
        List[User]: A list of User objects.
    """
    return database["users"]

@app.post("/users", tags=["users"])
def add_user(user: User) -> dict:
    """
    Add a new user to the library.

    Args:
        user (User): The User object to be added.

    Returns:
        dict: A message confirming the addition of the user.
    """
    database["users"].append(user)
    return {"message": "User added successfully"}
```

### Edge Case Handling and Input Validation

To handle edge cases such as invalid IDs and input validation, we utilize FastAPI's built-in validation and custom exception handling.

```python
from pydantic import BaseModel, Field

class BookBase(BaseModel):
    title: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    isbn: str = Field(..., min_length=10, max_length=13)

class BookCreate(BookBase):
    pass

class UserBase(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., regex=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

class UserCreate(UserBase):
    pass

# Update endpoints to use these models for validation
@app.post("/books", tags=["books"])
def add_book(book: BookCreate) -> dict:
    new_book = Book(**book.dict())
    database["books"].append(new_book)
    return {"message": "Book added successfully"}
```

### Performance Optimization

To optimize performance, we could use a dictionary for the in-memory database to allow O(1) lookups by ID.

```python
database = {
    "books": {},  # Dictionary for books indexed by ID
    "users": {},  # Dictionary for users indexed by ID
}

# Update the CRUD operations to use the new structure
@app.get("/books", tags=["books"])
def get_books() -> List[Book]:
    return list(database["books"].values())
```

## Documentation

Generate automatic documentation using Sphinx and Google-style docstrings.

```bash
sphinx-apidoc -o docs/source/ your_project_directory
sphinx-build -b html docs/source/ docs/build/html
```

## Practice Exercises

1. Add validation for ISBN uniqueness in books.
2. Implement pagination for the GET endpoints.
3. Add an endpoint to check out a book by a user.

## Key Takeaways and Summary

- **Comprehensive Logging and Error Handling**: Essential for monitoring and debugging.
- **Edge Cases and Input Validation**: Protect your API from invalid and malicious inputs.
- **Performance Optimization**: Always consider the efficiency of your data structures and algorithms.
- **Type Hints and Docstrings**: Improve code clarity and facilitate automatic documentation generation.
- **Best Practices**: Follow PEP 8 and community best practices for maintainable and scalable code.

By completing this capstone project, you've demonstrated your ability to design and implement a complex application using modern Python practices. Keep refining and expanding your application to further enhance your skills.
