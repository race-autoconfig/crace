class CraceError(Exception):
    """Exceptions for crace"""

    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass


class OptionError(CraceError):
    """Exception for option type """
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass


class ParameterDefinitionError(CraceError):
    """Exception for option type """
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass


class ParameterValueError(CraceError):
    """Exception for option type """
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass


class ModelError(CraceError):
    """Exception for option type """
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass


class FileError(CraceError):
    """ Errors related to files (not found or invalid"""
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass


class CraceExecutionError(CraceError):
    """
    Errors related to the execution of jobs
    """
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass

class ExitError(CraceError):
    """Exit without error"""
    def __init__(self, message="No message provided."):
        super().__init__(message)
    pass