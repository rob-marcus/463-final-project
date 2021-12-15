"""Defines various image types used during processing, importing, exporting.
"""
from enum import Enum
from os.path import basename


class FileExtensions(Enum):
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    HDR = "hdr"
    TIFF = "tiff"

    @classmethod
    def __has_value(cls, value: str):
        """Returns a dict of enums with the stringy file extensions as the key,
        and the enum as the value.
        :param value some value to check membership for, i.e., 'jpeg'.
        """
        return value in cls._value2member_map_

    @classmethod
    def path_to_enum(cls, filepath: str):
        """Gets the enum value for a given file path, if it exists.

        ext_str should not be prefixed with a period.
        :param filepath: a given string filepath terminated with a file
        extension, i.e., ../../myfile.jpg would return FileExtensions.JPG
        :return: if ext_str is a valid file extension, will return the
            corresponding enum field. Else, None.
        """
        ext = basename(filepath).split(".")[-1]

        return FileExtensions(ext) if cls.__has_value(ext) else None

    def __repr__(self):
        return self.value
