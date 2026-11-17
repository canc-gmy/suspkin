"""Study runner and report builder (tables in, files out)."""

from suspkin.report.builder import build_report
from suspkin.report.studies import StudyResults, run_standard_studies

__all__ = ["StudyResults", "build_report", "run_standard_studies"]
