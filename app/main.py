"""
Main application entry point
"""

from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title="Agent Mark1",
    description="Marketing Analytics Agent with RAG capabilities",
    version="1.0.0"
)


def main():
    """Main entry point for the application."""
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)


if __name__ == "__main__":
    main()
