# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Entry point for the auDeep application"""
import argparse
import logging
import os
import sys
import pkg_resources
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from cliff.app import App
from cliff.commandmanager import CommandManager


class _VersionAction(argparse.Action):
    """
    A custom argparse Action, which preserves the formatting of the version string.
    """

    def __init__(self,
                 option_strings,
                 version=None,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help="show program's version number and exit"):
        super(_VersionAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)
        self.version = version

    def __call__(self, parser, namespace, values, option_string=None):
        version = self.version
        if version is None:
            version = parser.version
        formatter = argparse.RawTextHelpFormatter(prog=parser.prog)
        formatter.add_text(version)
        parser._print_message(formatter.format_help(), sys.stdout)
        parser.exit()


class AuDeepApp(App):
    """
    Main class of the auDeep application.
    """

    def __init__(self):
        super(AuDeepApp, self).__init__(
            description="Deep Representation Learning Toolkit for Acoustic Data",
            version=pkg_resources.require("audeep")[0].version,
            command_manager=CommandManager("audeep.commands"),
            deferred_help=True
        )

    def build_option_parser(self, description, version, argparse_kwargs=None):
        argparse_kwargs = argparse_kwargs or {}
        argparse_kwargs["conflict_handler"] = "resolve"

        parser = super().build_option_parser(description, version, argparse_kwargs=argparse_kwargs)

        version_str = "%(prog)s {0}  Copyright (C) 2017-2021 Michael Freitag, Shahin Amiriparian, Maurice Gerczuk, Sergey Pugachevskiy, Nicholas Cummins, " \
                      "Bjoern Schuller\n" \
                      "License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>.\n" \
                      "This is free software: you are free to change and redistribute it.\n" \
                      "There is NO WARRANTY, to the extent permitted by law.".format(version)

        parser.add_argument("--version",
                            action=_VersionAction,
                            version=version_str)

        return parser

    def initialize_app(self, argv):
        rootLogger = logging.getLogger("")
        rootLogger.handlers[0].setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))

        # suppress tensorflow logging if the --debug option is not set
        if not self.options.debug:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

        self.LOG.debug("initializing app")

    def prepare_to_run_command(self, cmd):
        self.LOG.debug('prepare_to_run_command %s', cmd.__class__.__name__)

    def clean_up(self, cmd, result, err):
        self.LOG.debug('clean_up %s', cmd.__class__.__name__)


def main(argv=sys.argv[1:]):
    myapp = AuDeepApp()
    return myapp.run(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
