# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Utilities for ease of use."""


import tempfile


__all__ = ('make_settings_file',)


def make_settings_file(settings):
    """Create a temporary EMRecon settings file.

    Parameters
    ----------
    settings : `dict`
        Dictionary with the settings that should be written to the file.
        Consult EMRecon doc for information on what options are valid.
    """
    settings_str = '\n'.join('{}={}'.format(key, settings[key])
                             for key in settings)

    with tempfile.NamedTemporaryFile(delete=False, suffix='emrecon') as settings_file:
        settings_file.write(settings_str)
        settings_file_name = settings_file.name

    return settings_file_name
