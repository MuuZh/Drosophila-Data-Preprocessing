"""Module entry point to run the ΔF/F0 processor via `python -m deltaf_processor`."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
