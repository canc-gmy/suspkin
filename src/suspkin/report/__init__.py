"""Study runner, report builder and result bundles (tables in, files out)."""

from suspkin.report.builder import build_report
from suspkin.report.bundle import (
    BUNDLE_SCHEMA,
    Bundle,
    BundleError,
    bundle_to_results,
    is_bundle,
    load_bundle,
    save_bundle,
)
from suspkin.report.studies import StudyResults, run_standard_studies

__all__ = [
    "BUNDLE_SCHEMA",
    "Bundle",
    "BundleError",
    "StudyResults",
    "build_report",
    "bundle_to_results",
    "is_bundle",
    "load_bundle",
    "run_standard_studies",
    "save_bundle",
]
