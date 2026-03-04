import logging
from sys import platform
from warnings import warn

# Configure Logging
glaes_logger = logging.getLogger("GLAES")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class GlaesError(Exception):
    pass


class GlaesMultiProcessingWarning(Warning):
    multi_processing_warning_message = (
        "Multiprocessing has been set to 'False' because it is not available for Windows or Mac."
        " To deactivate this warning, please set the multiProcess variable to False. On Windows and "
        "Mac, new processes must be spawned, which requires the serialisation of the method to be "
        "executed via multiprocessing. However, Geokit contains objects that cannot be serialised by Pickle. "
        "On Linux, however, new processes are inherited and no serialisation is required."
    )


def checkMultiProcessingAvailability(multiProcess: bool) -> bool:
    """Multiprocessing is not available on all operating systems. If the user wants to
    to use multiprocessing on an unsupported operating system, multiprocessing will be
    deactivated and a warning appears.

    Parameters
    ----------
    multiProcess : bool
        A flag indicating whether multiprocessing should be used as indicated by the user.
        If multiprocessing is not available for the operating system, multiprocessing will be deactivated and a warning appears.

    Returns
    -------
    bool
        The corrected value for multiprocessing availability.
    """
    if platform == "linux" or platform == "linux2":
        multiProcessCorrected = multiProcess
    elif platform == "darwin" and multiProcess is True:
        multiProcessCorrected = False
        warn(message=GlaesMultiProcessingWarning.multi_processing_warning_message, category=GlaesMultiProcessingWarning)
    elif platform == "win32" and multiProcess is True:
        multiProcessCorrected = False
        warn(message=GlaesMultiProcessingWarning.multi_processing_warning_message, category=GlaesMultiProcessingWarning)
    elif platform == "darwin" and multiProcess is False:
        multiProcessCorrected = multiProcess
    elif platform == "win32" and multiProcess is False:
        multiProcessCorrected = multiProcess
    else:
        multiProcessCorrected = False
        warn(message=GlaesMultiProcessingWarning.multi_processing_warning_message, category=GlaesMultiProcessingWarning)
    return multiProcessCorrected
