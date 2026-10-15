"""``python -m igris_c_teleop`` entry point."""

from .app.producer import main

if __name__ == "__main__":
    raise SystemExit(main())
