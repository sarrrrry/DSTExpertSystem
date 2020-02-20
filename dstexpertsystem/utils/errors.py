from pathlib import Path


def is_exists_path(path):
    return Path(path).exists()


class Errors:
    LINE = "=" * 25
    BASE_MSG = "\n{line}\n".format(line=LINE)

    def __call__(self, msg, exception):
        return exception(msg)

    @classmethod
    def FileNotFound(self, path: Path):
        path = Path(path)
        msg = self.BASE_MSG
        msg += "NOT Exists Path:\n"

        path_gradually = Path(path.parts[0])
        for path_part in path.parts[1:]:
            path_gradually /= path_part
            msg += "\tExists: {}, {}\n".format(path_gradually.exists(), path_gradually)

        return FileNotFoundError(msg)

    @classmethod
    def GlobError(cls, path, suffix):
        msg = cls.BASE_MSG
        msg += "glob length is 0:\n"
        msg += "\tpath   : {}\n".format(path)
        msg += "\tsuffix : {}\n".format(suffix)

        return FileNotFoundError(msg)

    @classmethod
    def ValueError(self, msg):
        err_msg = self.BASE_MSG
        err_msg += msg
        return ValueError(err_msg)
        
