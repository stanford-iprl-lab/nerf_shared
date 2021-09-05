import bpy
import json
from mathutils import Matrix


#filepath = bpy.path.abspath('//..') + "/playground_testing.json"

#print( bpy.path.abspath('//') + 'scenes/objects/sphere/scene.gltf')

#marker = bpy.ops.import_scene.gltf(filepath=  bpy.path.abspath('//') + 'scenes/objects/sphere/scene.gltf')

#bpy.data.objects["orb"] = marker

bpy.ops.mesh.primitive_ico_sphere_add(location=(5.,5.,0.), radius=0.01)

object_to_copy = bpy.data.objects["Icosphere"]

path_parent = object_to_copy.copy()
path_parent.name = "PATH"


collection = bpy.data.collections.new("Diag_fast")
bpy.context.scene.collection.children.link(collection)

collection.objects.link(path_parent)

import os.path

save = 0
while True:
    filepath = bpy.path.abspath('//paths/') +'drone_pose' + f'{100*save}.json'
    if not os.path.isfile(filepath):
        break

    bpy.context.scene.frame_set(save)

    print("LOADING SAVE #"+str(save))
    with open(filepath) as f:
        meta = json.load(f)
        poses = meta['poses']
        for i, pose in enumerate(poses):
            name = "path_"+str(i)
            if save == 0:
                new_object = object_to_copy.copy()
                collection.objects.link(new_object)

                new_object.name = name 
                new_object.parent = path_parent
                new_object.matrix_parent_inverse = path_parent.matrix_world.inverted()
            else:
                new_object = bpy.data.objects[name]
                
            new_object.matrix_world = Matrix(pose)

            new_object.keyframe_insert(data_path="location")
            if new_object.rotation_mode == "QUATERNION":
                new_object.keyframe_insert(data_path = 'rotation_quaternion')
            else:
                new_object.keyframe_insert(data_path = 'rotation_euler')
            
    bpy.context.view_layer.update() 
    save += 1