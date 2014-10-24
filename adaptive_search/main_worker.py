from __future__ import print_function
import sys
import yaml
import argparse

from .config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to worker configuration file (yaml)')
    parser.parse_args()
    parser.set_defaults(func=execute)
    args = parser.parse_args()
    args_func(args, parser)

        
def execute(args, parser):
    config = Config(args.config)
    config.check_fields
    print(config.estimator())
    print(config.param_grid())
    print(config.cv())
    print(config.dataset())
    print(config.experiments())
    
    
def args_func(args, p):
    try:
        args.func(args, p)
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        if e.__class__.__name__ not in ('ScannerError', 'ParserError'):
            message = """\
An unexpected error has occurred, please consider sending the
following traceback to the mixtape GitHub issue tracker at:

        https://github.com/rmcgibbo/mixtape/issues

"""
            print(message, file=sys.stderr)
        raise  # as if we did not catch it
