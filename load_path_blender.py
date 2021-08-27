

import bpy
import json
from mathutils import Matrix


#filepath = bpy.path.abspath('//..') + "/playground_testing.json"


save = 3
filepath = bpy.path.abspath('//../paths/') +str(save)+"_testing.json"
object_to_copy = bpy.data.objects["Cube"]


path_parent = object_to_copy.copy()
path_parent.name = "PATH"

bpy.context.collection.objects.link(path_parent)

with open(filepath) as f:
    for i, line in enumerate(f):

        new_object = object_to_copy.copy()
        bpy.context.collection.objects.link(new_object)

        pose = json.loads(line)
        new_object.matrix_world = Matrix(pose)
        print(i, pose)

        new_object.name = "path_"+str(i)
        new_object.parent = path_parent
        new_object.matrix_parent_inverse = path_parent.matrix_world.inverted()
        
#bpy.context.scene.update()
bpy.context.view_layer.update() 