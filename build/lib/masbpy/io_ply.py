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

def read_ply(infile, limit_points=0, move_to_origin=False, read_normals=True):
	"""collect vertex coordinates and normals from input file"""
	ox,oy,oz = (0,0,0)
	datadict = {}
	with open(infile) as f:
		vertexcount = facecount = None
		property_dict = {}
		property_count = 0
		while True:
			line = f.readline()
			if line.startswith("element vertex"):
				vertexcount = line.split()[-1]
				while True:
					line = f.readline()
					prev_cursor_pos = f.tell()
					if line.startswith("property"):
						property_name = line.split()[-1]
						property_dict[property_name] = property_count
						property_count += 1
					else:
						f.seek(prev_cursor_pos)
						break
			if line.startswith("element face"):
				facecount = line.split()[-1]
			if line.startswith("end_header"):
				break

		datadict['coords'] = []
		datadict['normals'] = []
		if limit_points:
			vertexcount = limit_points
		for i in range(int(vertexcount)):
			line = f.readline()

			
			line_ = line.split()
			x,y,z = line_[property_dict['x']], line_[property_dict['y']], line_[property_dict['z']]
			if read_normals:		
				nx,ny,nz = line_[property_dict['nx']], line_[property_dict['ny']], line_[property_dict['nz']]
			
			if move_to_origin and i==0:
				ox,oy,oz = float(x), float(y), float(z)
			
			datadict['coords'].append(np.array([float(x)-ox,float(y)-oy,float(z)-oz]))
			if read_normals: datadict['normals'].append(np.array([float(nx),float(ny),float(nz)]))

		if facecount is not None:
			datadict['faces'] = []
			for i in range(int(facecount)):
				line = f.readline().split()
				vertex_ids = [int(x) for x in line[1:]]
				datadict['faces'].append(vertex_ids)

	datadict['coords'] = np.array(datadict['coords'], dtype=np.float32)
	datadict['normals'] = np.array(datadict['normals'], dtype=np.float32)
	return datadict