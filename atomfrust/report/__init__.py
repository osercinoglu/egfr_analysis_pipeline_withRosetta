"""Confound-aware reporting (plan D7).

The layer between a finished analysis and a claim. Its one rule: a correlation is emitted as
a triple — raw, partial controlling ``n_contacts_total``, and the OLS coefficient with its
VIF — and :func:`~atomfrust.report.collect.render_report` refuses to print a headline for any
descriptor whose raw CI excludes zero while its partial CI does not. That is structural, not
a convention: plan §2.1 shows the pocket-size confound is *derived* from the published
many-body formula, so the reporting layer has to be unable to hide it.

:mod:`~atomfrust.report.plots` is imported lazily by ``render_report`` so that importing this
package does not require matplotlib.
"""

from atomfrust.report.collect import (
    COVARIATE_WARNING,
    DEFAULT_COVARIATE,
    HEADLINE_MARK,
    collect_analyses,
    correlation_triple,
    default_descriptors,
    headline_is_permitted,
    render_report,
    report_table,
    resolve_descriptor,
)

__all__ = [
    "COVARIATE_WARNING",
    "DEFAULT_COVARIATE",
    "HEADLINE_MARK",
    "collect_analyses",
    "correlation_triple",
    "default_descriptors",
    "headline_is_permitted",
    "render_report",
    "report_table",
    "resolve_descriptor",
]
