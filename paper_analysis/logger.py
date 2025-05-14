from loguru import logger

# Configure the logger to write logs to a file
logger.add("application.log", rotation="10 MB", retention="7 days", level="INFO")

# Export the logger for use in other scripts
__all__ = ["logger"]