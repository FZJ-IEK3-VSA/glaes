import logging

# Configure Logging
glaes_logger = logging.getLogger("GLAES")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class GlaesError(Exception):
    pass
