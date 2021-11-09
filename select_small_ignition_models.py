import pybullet as p
from typing import List
import random
import os
from draw_aabb import drawAABB
import time
from shutil import copyfile
from distutils.dir_util import copy_tree
from pathlib import Path
p.connect(p.GUI)

Path("small_object_models/").mkdir(parents=True,exist_ok=True)

objects_list: List = os.listdir('object_models')

for obj in objects_list:
    id = p.loadSDF('object_models/' + obj + '/1/model.sdf')[0]
    # p.resetBasePositionAndOrientation(id, [random.uniform(-1, 1),
    #                                        random.uniform(-1, 1),
    #                                        0.02],
    #                                   p.getQuaternionFromEuler([0, 0, 0]))
    aabb=p.getAABB(id)
    # print(aabb[0])
    # print(aabb[1])
    #
    # drawAABB(aabb)
    # time.sleep(5)

    aabb_min=aabb[0]
    aabb_max=aabb[1]

    for i in range(3):
        if (aabb_max[i]-aabb_min[i])<0.09:
            copy_tree("object_models/"+obj,"small_object_models/"+obj)
            break

    p.removeBody(id)




