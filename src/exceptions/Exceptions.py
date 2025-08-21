class MissingEnvironmentVariableError(Exception):
    def __init__(self, message: str = "Some env variables are missing"):
        super().__init__(message)
