import os
from shutil import copyfile
names=os.listdir('object_models')

DEBUG=True

if DEBUG:
    print(names)
    print(len(names))

for name in names:
    copyfile('object_models/'+name+'/1/materials/textures/texture.png','object_models/'+name+'/1/meshes/texture.png')

