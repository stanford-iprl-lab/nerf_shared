

import bpy
import json
from mathutils import Matrix

import os.path

object_to_copy = bpy.data.objects["tim_drone"]

def load_MPC(experiment_name):
    pass

TYPE = "mpc"
#TYPE = "train"

def load_training(experiment_name):
    collection = bpy.data.collections.new(experiment_name +"_"+TYPE)
    bpy.context.scene.collection.children.link(collection)

    save = 0
    while True:
        filepath = bpy.path.abspath('//../experiments/' + experiment_name + '/'+TYPE+'/') +str(save)+".json"
        if not os.path.isfile(filepath):
            if save == 0:
                bpy.context.window_manager.popup_menu(lambda s,c:s.layout.label(text=""), title = "can't find first file", icon = "ERROR")
            break

        bpy.context.scene.frame_set(save)

        print("LOADING SAVE #"+str(save))
        with open(filepath) as f:
            data = json.load(f)

            for i in range(len(data['poses'])):
                name = "path_"+str(i)
                if save == 0:
                    new_object = object_to_copy.copy()
                    collection.objects.link(new_object)

                    new_object.name = name 
                else:
                    new_object = bpy.data.objects[name]

                pose = data['poses'][i]
                new_object.matrix_world = Matrix(pose)
                print(i, pose)

                new_object.keyframe_insert(data_path="location")
                if new_object.rotation_mode == "QUATERNION":
                    new_object.keyframe_insert(data_path = 'rotation_quaternion')
                else:
                    new_object.keyframe_insert(data_path = 'rotation_euler')
                
        bpy.context.view_layer.update() 
        save += 1

def get_endpoints():
    s_loc = [round(x,2) for x in bpy.data.objects['Start'].location]
    bpy.data.objects['Start'].location = Vector(s_loc)
    e_loc = [round(x,2) for x in bpy.data.objects['End'].location]
    bpy.data.objects['End'].location = Vector(e_loc)
    s = ""
    s += 'start_pos = torch.tensor(' + str( s_loc ) +")"
    s += "\n"
    s += 'end_pos = torch.tensor(' + str( e_loc ) +")"
    s += "\n"
    if False:
        s += 'start_R =' + bpy.data.objects['Start'].matrix_world.__repr__().replace("Matrix","torch.tensor") + "[:3,:3]"
        s += "\n"
        s += 'end_R ='   + bpy.data.objects['End'].matrix_world.__repr__().replace("Matrix","torch.tensor") + "[:3,:3]"
        s += "\n"
    print(s)



experiment_name = "stonehenge_astar"

load_training(experiment_name)
