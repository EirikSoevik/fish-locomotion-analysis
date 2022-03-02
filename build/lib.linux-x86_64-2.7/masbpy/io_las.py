# This file is part of masbpy.

# masbpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# masbpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with masbpy.  If not, see <http://www.gnu.org/licenses/>.

# Copyright 2015 Ravi Peters

import numpy as np

def read_las(infile, keys=None):
	try:
		from laspy.file import File
	except ImportError:
		print("Cannot read las files without laspy module")
		raise

	inFile = File(infile)
	datadict = {}
	datadict['coords'] = np.column_stack([ np.array(a, dtype=np.float32) for a in [inFile.x, inFile.y, inFile.z] ])
	return datadict

