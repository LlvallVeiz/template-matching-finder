"""Custom exceptions for template-matching-finder."""

from __future__ import annotations


class TemplateMatchingError(Exception):
    """Base class for all template-matching errors."""


class TemplateTooLargeError(TemplateMatchingError):
    """Raised when the template is larger than the search image in any dimension."""


class EmptyImageError(TemplateMatchingError):
    """Raised when an image or template array is empty."""


class InvalidMethodError(TemplateMatchingError):
    """Raised when an unknown matching method name is requested."""


class InvalidParameterError(TemplateMatchingError):
    """Raised when a parameter value is out of its valid range."""
