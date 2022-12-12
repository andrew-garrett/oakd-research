import logging

class CustomFormatter(logging.Formatter):
	"""Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

	def __init__(self):
		super().__init__()
		self.grey = '\x1b[38;21m'
		self.blue = '\x1b[38;5;39m'
		self.yellow = '\x1b[38;5;226m'
		self.red = '\x1b[38;5;196m'
		self.bold_red = '\x1b[31;1m'
		self.reset = '\x1b[0m'
		self.fmt = "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - (%(filename)s:%(lineno)d) - %(message)s"
		self.FORMATS = {
			logging.DEBUG: self.grey + self.fmt + self.reset,
			logging.INFO: self.blue + self.fmt + self.reset,
			logging.WARNING: self.yellow + self.fmt + self.reset,
			logging.ERROR: self.red + self.fmt + self.reset,
			logging.CRITICAL: self.bold_red + self.fmt + self.reset
		}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)
