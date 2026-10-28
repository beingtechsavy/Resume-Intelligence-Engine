"""Local HTTP API entry point — thin shim over the rie.web package.

The implementation lives in ``rie.web.server`` (HTTP routing), ``rie.web.store``
(job registry), ``rie.web.runner`` (generation worker), and ``rie.data``
(profile loading).  This module keeps the historical ``python app.py`` command
and the test-suite imports working.
"""

from __future__ import annotations

from rie.data import (  # noqa: F401 - re-exported for backwards compatibility
    load_profile,
    parse_json_body,
    profile_summary,
)
from rie.web.runner import restore_completed_runs  # noqa: F401
from rie.web.server import build_server, main  # noqa: F401
from rie.web.store import APIError, JobStore, JobStoreFullError  # noqa: F401

if __name__ == "__main__":
    main()
