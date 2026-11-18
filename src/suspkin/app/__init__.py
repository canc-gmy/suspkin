"""Experimental local web app backend (FastAPI) for viewing and comparing designs.

Needs the ``app`` extra (``uv sync --extra app``). The API is versioned under ``/api/v1`` and
may change before it is declared stable. Importing this package does not import FastAPI;
only :mod:`suspkin.app.server` and :mod:`suspkin.app.routes` do.
"""
