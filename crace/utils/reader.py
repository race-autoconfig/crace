import os
import re
import mimetypes
from crace.errors import FileError, OptionError


class Reader:
    """
    Class that contains the methods for reading text files processed by crace
    """

    @staticmethod
    def check_readable(path):
        """
        Check if the file is valid or readable. Otherwise, raise an exception.
        """
        if mimetypes.guess_type(path)[0] != 'text/plain' and not \
                os.access(path, os.R_OK):
            raise FileError(f"{path} is not readable.")
        return True

    @staticmethod
    def check_dir_readable(path):
        """
        Check if the directory is valid or readable. Otherwise, raise an exception.
        """
        if not os.access(path, os.R_OK):
            raise FileError(f"{path} is not readable.")
        return True

    @staticmethod
    def get_matched_line(
            line: str, patterns: list, separator=r"\S+", delimited=False):
        """
        Read a line and splits it into two parts: 
            the first is a string obtained from the beginning of the line that matches the given pattern, 
            and the second is the rest of the line without the match. 
        If no match is found with the pattern in the first series of characters, the first element is None."
        """
        value = None
        replaced = ""
        words = re.findall(separator, line)
        for pattern in patterns:
            if re.match(pattern, words[0]):
                value = replaced = words[0]
                if delimited:
                    value = value[1:len(value) - 1]
        return value, line.replace(replaced, "", 1).strip()

    @staticmethod
    def get_readable_lines(path: str):
        """
        Read a file and retrieve the uncommented lines.
        """
        data = open(path, "r")
        lines = data.read().split("\n")
        data.close()
        lines = [
            l.strip() for l in lines
            if len(re.sub("#.*$", "", l)) > 1 and len(l) > 1
        ]
        return lines

    @staticmethod
    def replace_logical_operator(condition: str):
        """
        Receive a condition in the form of a string and 
        attempt to replace R expressions to make it executable.
        """
        value = condition
        if condition[0] == '!':
            value = "not" + condition[2:len(condition) - 1]
        value = value.replace("%in%", "in").replace("c(", "("). \
            replace("&", "and").replace("|", "or")

        return value

    @staticmethod
    def check_executable_cmdline(cmd: str, file: str):
        """
        Check if the provided command line for executing the file is valid.

        :param cmd: The command line to execute the file (e.g., "Rscript file.r")
        :param file: The file to be executed
        :return: boolean if valid.
        """
        import shutil
        from pathlib import Path

        # check if cmd exist in PATH
        cmd_path = shutil.which(cmd)
        if cmd_path is None:
            raise OptionError(f"Provided command '{cmd}' not found in PATH")

        # check if file exists
        # this file could be only a file without execution permission
        # if provided together with cmd
        file_path = Path(file)
        if not file_path.exists():
            raise OptionError(f"Provided file '{file}' not found")

        return True