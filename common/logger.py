import logging
import sys
from typing import Optional

def setup_logging(
	level: str = "info",
	log_format: Optional[str] = None,
) -> None:
	"""
	Configure logging for the entire application.

	Args:
		level: Logging level (e.g., "debug", "info").
		log_format: Optional custom format string. If None, a default format is used.
	"""

	# Convert level to proper logging constant
	if level == "info":
		level = logging.INFO
	if level == "debug":
		level = logging.DEBUG
	if level == "off":
		level = logging.NOTSET
	
	if log_format is None:
		log_format = "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"

	logging.basicConfig(
		level = level,
		format = log_format,
		datefmt = '%H:%M:%S',
		handlers = [logging.StreamHandler(sys.stdout)],
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