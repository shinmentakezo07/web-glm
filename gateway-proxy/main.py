"""Entry point — delegates to server.app so `uv run main.py` works."""
from server import app  # noqa: F401

if __name__ == "__main__":
    from server import main  # runs uvicorn
    main()
