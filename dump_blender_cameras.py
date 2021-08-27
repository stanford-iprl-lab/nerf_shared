

import bpy
import json
from mathutils import Matrix


# filepath = "/output/path/camera_poses.json"

# bpy.path.abspath('//') is the directory where the blend file is saved
filepath = bpy.path.abspath('//..') + "/camera_poses.json"


# save the 4x4 poses of all object with "Camera" in their name
with open(filepath, "w+") as f:
    for obj in bpy.data.objects:
        if "Camera" not in obj.name:
            continue

        pose = [list(row) for row in obj.matrix_world]
        json.dump(pose, f)
        f.write("\n")

