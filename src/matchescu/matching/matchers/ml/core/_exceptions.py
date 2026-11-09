from os import PathLike


class ModelLoadingError(Exception):
    def __init__(self, path: str | PathLike, message: str | None = None) -> None:
        message = message or "failed to load model from path"
        super().__init__(f"{path}: {message}")
        self.path = path
