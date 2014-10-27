# # STARTUP SPEED DEBUGGING
# # enabling this will log every import that's being made
# import sys
# class CustomImporter(object):
#     def find_module(self, fullname, path):
#         print(fullname, path)
#         return None
# sys.meta_path.append(CustomImporter())

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
