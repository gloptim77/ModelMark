import logging
import sys
from typing import Optional

def setup_logging(log_format: Optional[str] = None) -> None:
	"""
	Configure logging for the entire application.

	Args:
		log_format: Optional custom format string. If None, a default format is used.
	"""

	if log_format is None:
		log_format = "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"

	logging.basicConfig(
		level = logging.DEBUG,
		filemode = 'w',
		filename='modelmark_files/modelmark.log',
		format = log_format,
		datefmt = '%H:%M:%S',
	)

def get_logger(name: str) -> logging.Logger:
	"""
	Convenience function to get a logger for a module.

	Args:
		name: Usually __name__.

	Returns:
		A configured logger instance.
	"""
	return logging.getLogger(name)