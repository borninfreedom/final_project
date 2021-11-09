import pybullet as p
import os
import time
p.connect(p.DIRECT)

names=os.listdir('small_object_models')

for name in names:
    id=p.loadSDF('small_object_models/'+name+'/1/model.sdf')[0]
    print(p.getBasePositionAndOrientation(id))
    time.sleep(1)
    p.removeBody(id)

