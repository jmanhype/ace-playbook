Operations Runbook
==================

Quick reference guide for operating and troubleshooting ACE Playbook in production.

Health Checks
-------------

.. code-block:: bash

   # Check service status
   curl http://localhost:8000/health

   # Expected response:
   {
     "status": "healthy",
     "checks": {
       "database": "ok",
       "faiss": "ok",
       "embedding_service": "ok"
     },
     "timestamp": "2025-01-15T10:30:00Z"
   }

For comprehensive troubleshooting, see the full runbook documentation.
