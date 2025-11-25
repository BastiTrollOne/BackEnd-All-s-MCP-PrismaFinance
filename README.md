# PrismaFinance Backend

This project is the backend for PrismaFinance, a FastAPI application that integrates with LangChain, LangGraph, and Metabase.

## Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    uvicorn app.main:app --reload
    ```
