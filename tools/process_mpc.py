
import pathlib
import json
import os

# os.chdir("/Users/mik/Downloads/playground_filter_wind")
os.chdir("/Users/mik/Downloads/stonehenge")

exec_graph = pathlib.Path("execute_graph")
exec_poses = pathlib.Path("execute_poses")

for file in exec_poses.iterdir():
    print(file.name)
    if file.name[0] != "4":
        file.unlink()

print("renaming exec_poses")
for file in exec_poses.iterdir():
    print(file.name)
    file.rename(exec_poses / file.name[6:])


print("unlinking exec_graph")
for file in exec_graph.iterdir():
    print(file.name)
    if file.name[0] != "4":
        file.unlink()

print("renaming exec_graph")
for file in exec_graph.iterdir():
    file.rename(exec_graph / file.name[6:])


max_step = max(int(file.name.split(".")[0]) for file in exec_graph.iterdir())

mpc = pathlib.Path("mpc")
mpc.mkdir()

for i in range(max_step + 1):
    # poses = []
    # with open(exec_poses / f"{i}.json","w+") as f:
    #     for line in f.readlines():
    #         poses.append( json.loads(line))

    poses = json.load( open(exec_poses / f"{i}.json") )['poses']
    g = json.load( open(exec_graph / f"{i}.json") )
    g['poses'] = poses
    with open(mpc / f"{i}.json","w+") as f:
        json.dump(g, f)


train_graph = pathlib.Path("train_graph")
train_poses = pathlib.Path("train_poses")

train = pathlib.Path("train")
train.mkdir()

max_step = max(int(file.name.split(".")[0]) for file in train_graph.iterdir())

for i in range(max_step + 1):
    # poses = []
    # with open(train_poses / f"{i}.json","w+") as f:
    #     for line in f.readlines():
    #         poses.append( json.loads(line))

    poses = json.load( open(train_poses / f"{i}.json") )['poses']
    g = json.load( open(train_graph / f"{i}.json") )
    g['poses'] = poses
    with open(train / f"{i}.json","w+") as f:
        json.dump(g, f)

