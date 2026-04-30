# services/listener/__init__.py
"""
Module listener — Traitement event-driven des fichiers JSON Power Automate.

Expositions publiques :
    FileListener — classe principale à instancier dans le scheduler APScheduler.

Usage :
    from services.listener import FileListener

    listener = FileListener()
    scheduler.add_job(listener.process_pending_requests, "interval", seconds=30)
    scheduler.add_job(listener.process_pending_decisions, "interval", seconds=15)
"""
from services.listener.listener import FileListener

__all__ = ["FileListener"]
