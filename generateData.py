# to install any required packages with PIP into the blender python:
# e.g. /PATH/TO/BLENDER/python/bin$ ./python3.7m -m pip install pandas

import bpy
import bpy_extras
import os
import sys
import random
import math
import numpy as np
import json
import datetime
import shutil
import random
from mathutils import Vector, Matrix, Color
from pathlib import Path

sys.path.append(os.getcwd())
import config

cfg = config.cfg()


def saveCOCOlabel(images, annotations):
    # https://cocodataset.org/#format-data
    info = {
        "year": datetime.datetime.now().year,
        "version": "1.0",
        "description": "SynthBPSet: Synthetic Bike Parking Dataset created",
        "contributor": "xxxx xxxx",
        "url": "",
        "date_created": str(datetime.datetime.now()),
    }

    coco = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": [{"supercategory": "bike", "id": 0,"name": cfg.classes[0]},
                       {"supercategory": "bike", "id": 1,"name": cfg.classes[1]},
                       {"supercategory": "bike", "id": 2,"name": cfg.classes[2]}]
    }

    with open('./DATASET/' + cfg.out_folder +"/annotation_coco.json", "w") as write_file:
        json.dump(coco, write_file, indent=2)

texture_list = []
texture_ground_list = []
texture_list_grass = []
texture_list_inlcude_grass = []
texture_list_road = []
texture_list_road_and_grass = []
texture_list_building = []
texture_map_offset_x = []
texture_map_offset_y = []

def LoadData():
    image_list = os.listdir(cfg.object_texture_path)
    for img in image_list:    
        texture_ground_list.append(img)
        bpy.data.images.load(os.path.join(cfg.object_texture_path,img))

    texture_list_inlcude_grass.extend(texture_ground_list)
    grass_image_list = os.listdir(cfg.grass_texture_path)
    for img in grass_image_list:    
        texture_list_grass.append(img)
        bpy.data.images.load(os.path.join(cfg.grass_texture_path,img))

    texture_list_inlcude_grass.extend(texture_list_grass)
    road_image_list = os.listdir(cfg.road_texture_path)
    for img in road_image_list:    
        texture_list_road.append(img)
        bpy.data.images.load(os.path.join(cfg.road_texture_path,img))

    building_image_list = os.listdir(cfg.building_texture_path)
    for img in building_image_list:   
        texture_list_building.append(img)
        bpy.data.images.load(os.path.join(cfg.building_texture_path,img))

    texture_list_road_and_grass.extend(texture_list_road)
    texture_list_road_and_grass.extend(texture_list_grass)


def orderCorners(objBB):
    """change bounding box corner order."""
    # change bounding box order according to
    # https://github.com/Microsoft/singleshotpose/blob/master/label_file_creation.md
    out = []
    corners = [v[:] for v in objBB]  # list of tuples (x,y,z)
    out.append(corners[0])  # -1 -1 -1
    out.append(corners[1])  # -1 -1 1
    out.append(corners[3])  # -1 1 -1
    out.append(corners[2])  # -1 1 1
    out.append(corners[4])  # 1 -1 -1
    out.append(corners[5])  # 1 -1 1
    out.append(corners[7])  # 1 1 -1
    out.append(corners[6])  # 1 1 1
    return out

def project_by_object_utils(cam, point):
    """returns normalized (x, y) image coordinates in OpenCV frame for a given blender world point."""
    scene = bpy.context.scene
    # print(type(scene), type(cam), type(point))
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x, 1 - co_2d.y))  # normalized


def cameraAugment():
    """sample x,y,z on sphere and place camera (looking at the origin)."""
    # Random place the camera
    if cfg.camPathEnableV:
        bpy.context.scene.objects['cameraBox'].constraints['Follow Path'].offset = float(random.randint(cfg.camPathVa,cfg.camPathVb)) 
    else:
        bpy.context.scene.objects['cameraBox'].constraints['Follow Path'].offset = -70.0
    
    if cfg.camPathEnableH:
        bpy.context.scene.objects['cameraCircleBox'].constraints['Follow Path'].offset = float(random.randint(cfg.camPathHa,cfg.camPathHb)) 
    else:
        bpy.context.scene.objects['cameraCircleBox'].constraints['Follow Path'].offset = 0.0 


    bpy.context.view_layer.update()
    return 

def lightAugment():
    # Sun path (location)
    bpy.context.scene.objects['Point'].constraints['Follow Path'].offset = random.random() * -100
    bpy.data.lights['Point'].energy = random.randint(100000,180000)
    # change world brightness (dynamic sky)
    randValue = random.uniform(cfg.worldStringth_min, cfg.worldStringth_max)
    bpy.data.worlds["Dynamic_1"].node_tree.nodes["Scene_Brightness"].inputs[1].default_value = randValue
    return


allocatedLocations = []
yPathsRight = [13,20,28,36]
yPathsLeft = [55,63,71,79]
yPathsPedestrian = [-9,-12,-15]
yPathsBinsAndUmbrella = [-1,-3,-5]
vehicleBoundary = (30,2)

def placeObject(obj, yPaths, boundaries):
    '''
    obj: the object to be placed 
    yPaths: available locations in y axis
    boundaries: boundaries of the object (car)
    '''
    if 'Large' in obj.name:
        boundaries = (60,3)
                
    notAllocated = True
    while notAllocated:
        randY = random.sample(yPaths,1)[0]
        randX = random.randrange(-240,240,1)
        for (x,y) in allocatedLocations:
                if randX in list(range(x-boundaries[0], x+boundaries[0])) and randY in list(range(y-boundaries[1],y+boundaries[1])):
                    break
        notAllocated = False  # allocated
        
    obj.location[0], obj.location[1] = float(randX), float(randY)
    
    if 'Large' in obj.name:
        allocatedLocations.append((randX-20, randY))
        allocatedLocations.append((randX+20, randY))
    else:
        allocatedLocations.append((randX, randY))
    
    #change the color
    color = Color()
    hue = random.random() #* .1 # 0 - .2
    saturation = random.random() #* .7 # .2 - .8
    v = random.random()
    color.hsv = (hue,saturation, v)
    rgba = [color.r, color.g, color.b, 1]
    obj.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba

    if 'vehicleSmall' not in obj.name:
        randCarColor = random.sample(['white','black','clr'],1)
        #change the color
        color = Color()
        hue = random.random() #* .1 # 0 - .2
        saturation = random.random() #* .7 # .2 - .8
        v = random.random() * .3
        if randCarColor[0] == 'white':
            # print("white -----------")
            color.hsv = (0.0,0.0, 0.9)
        elif randCarColor[0] == 'clr':
            color.hsv = (hue,saturation, v)
        elif randCarColor[0] == 'black':
            # print("black -----------")
            color.hsv = (0.0,0.0, 0.0)
        else: 
            color.hsv = (hue,0.3, 0.5)

            

        rgba = [color.r, color.g, color.b, 1]
        obj.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba
        
        #random metallic
        obj.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.3,0.7)
        # random roughness
        obj.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[7].default_value = random.uniform(0.2,0.8)


def randomVehicles(cars,yPaths, boundary):

    for car in cars:
        placeObject(car, yPaths, boundary)
    

def poleAndUmbrella():
    pole = bpy.data.collections["pole"].objects
    for pl in pole:
        if pl.name == 'umbrella':
            #change the color
            color = Color()
            hue = random.random() #* .1 # 0 - .2
            saturation = random.random() #* .7 # .2 - .8
            color.hsv = (hue,saturation, 0.3)
            rgba = [color.r, color.g, color.b, 1]
            pl.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba
    
            while True:
                randX = random.randrange(-110,110,1)
                if not (randX > -30 and randX < 30):
                    break
        else:
            randX = random.randrange(-110,110,1)
        pl.location[0] = float(randX)  

 
def augmentBin(yPathsBin): 
    bins = bpy.data.collections["bin"].objects
    # Hide all bins
    for b in bins:
        bpy.context.view_layer.objects.active = b
        b.hide_render = True
        b.hide_viewport = True
            
    # plane = bpy.data.objects['groundPark']
    bin = bins[random.randint(0,len(bins)-1)]
    # print(bin)
    bin.hide_render = False
    bin.hide_viewport = False
    
    while True:
        randY = random.sample(yPathsBin,1)[0]
        randX = random.randrange(-60,60,1)
        if not (randX > -30 and randX < 30):
            break

        
    bin.location[0], bin.location[1] = float(randX), float(randY)
    
    # random resize the bin
    bpy.context.view_layer.objects.active = bin
    sz = random.uniform(cfg.minBinSize, cfg.maxBinSize)
    
    # first resize it to its default size
    bin.scale = Vector((0.911937,0.911937,0.380627))
    
    # then resize it randomly
    bin.scale = Vector((bin.scale[0] * sz, bin.scale[1] * sz, bin.scale[2] * sz))
    allocatedLocations.append((randX, randY))  

    # random rotate
    bin.rotation_euler[2] = math.radians(random.randint(-90,90))    


def augmentPedestrian():
    humanNum = 9 
    for i in range(humanNum):
        humanObjectName = 'human.Body.{0:03}'.format(i)
        h = bpy.data.collections["human"].objects[humanObjectName]
        randY = random.uniform(cfg.minPlaneY,cfg.maxPlaneY)
        randX = random.uniform(cfg.minPlaneX,cfg.maxPlaneX)

        h.location[0], h.location[1] = float(randX), float(randY)
        
        # random resize person
        sz = random.uniform(cfg.minSizePerson, cfg.maxSizePerson)
        
        # first resize it to its default size
        h.scale = Vector((1.0,1.0,1.0))
        
        # then resize it randomly
        h.scale = Vector((h.scale[0] * sz, h.scale[1] * sz, h.scale[2] * sz)) 

        # random rotate
        h.rotation_euler[2] = math.radians(random.uniform(0,360))    
        
    
def groundAugment(ground_list):
    for grnd in ground_list:
        # print(grnd)
        if (len(cfg.object_texture_path) > 0):
            mat = grnd.active_material
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]

            texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
            if grnd.name in ['groundBackLarge','groundBackSmall','groundMiddle']:
                textureImageIndex = random.randint(0,len(texture_list_inlcude_grass)-1)
                texImage.image = bpy.data.images[texture_list_inlcude_grass[textureImageIndex]]

                bpy.data.collections["vehiclesBack"].hide_render = True
                if 'grass' not in texture_list_inlcude_grass[textureImageIndex] and (grnd.name == 'groundBackLarge'):
                    augmentPedestrian()

                    bpy.data.collections["vehiclesBack"].hide_render = False
                    cars = bpy.data.collections["vehiclesBack"].objects 
                    idxShuffled = random.sample(list(range(len(cars))), len(cars))
                    # print(idxShuffled)
                    # random.shuffle(cars)
                    currentX = random.randrange(-50,-40)
                    for indx in idxShuffled:
                        # print(indx)
                        randOffsetX = random.randrange(8,13)

                        cars[indx].location[0] = float(currentX+randOffsetX)
                        currentX = currentX+randOffsetX
                        #change the color
                        color = Color()
                        hue = random.random() #* .1 # 0 - .2
                        saturation = random.random() #* .7 # .2 - .8
                        v = random.random()
                        color.hsv = (hue,saturation, v)
                        rgba = [color.r, color.g, color.b, 1]
                        cars[indx].material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba
                else:
                    bpy.data.collections["vehiclesBack"].hide_render = True

            elif grnd.name == 'groundRoadRight':
                # if grnd.name == 'groundRoadRight':
                    textureImageIndex = random.randint(0,len(texture_list_road_and_grass)-1)
                    texImage.image = bpy.data.images[texture_list_road_and_grass[textureImageIndex]]
                    if 'grass' not in texture_list_road_and_grass[textureImageIndex]:
                            bpy.data.collections["vehiclesRight"].hide_render = False
                            bpy.data.collections["vehiclesLeft"].hide_render = False

                            cars = bpy.data.collections["vehiclesRight"].objects 
                            randomVehicles(cars,yPathsRight,vehicleBoundary)

                            cars = bpy.data.collections["vehiclesLeft"].objects
                            randomVehicles(cars,yPathsLeft, vehicleBoundary)
                            
                            # augment pole and umbrella
                            poleAndUmbrella()
                    else:
                        bpy.data.collections["vehiclesRight"].hide_render = True
                        bpy.data.collections["vehiclesLeft"].hide_render = True
            else:
                textureImageIndex = random.randint(0,len(texture_ground_list)-1)
                texImage.image = bpy.data.images[texture_ground_list[textureImageIndex]]

            mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
            if grnd.data.materials:
                grnd.data.materials[0] = mat
            else:
                grnd.data.materials.append(mat)
                # texture_list

            if grnd.name == 'groundPavement':
                # scale the Bike's side Pavement length (X dimension)
                bpy.context.view_layer.objects.active = grnd
                grnd.scale[0] = random.randint(cfg.minSidePavementScaleX, cfg.maxSidePavementScaleX)
            else:
                # scale the walk side Pavement length (X dimension)
                bpy.context.view_layer.objects.active = grnd
                grnd.scale[0] = random.randint(cfg.minGroundScaleX, cfg.maxGroundScaleX)


def treeAugment(treeList):
    for tree in treeList:
        bpy.context.view_layer.objects.active = tree
        
        # random size
        if tree.name == 'treeSide':
            sz = random.uniform(cfg.minTreeSize,cfg.maxTreeSize)
            tree.scale = Vector((sz,sz,sz))
        else: # in case the tree on the far side 
            sz = random.uniform(cfg.minTreeSize,cfg.maxTreeSize)
            tree.scale = Vector((sz,sz,sz))

        bpy.context.object.modifiers['Array'].count = random.randint(10,22)

        # random tree array offset (Array modifier)
        bpy.context.object.modifiers['Array'].relative_offset_displace[0] = random.uniform(cfg.minTreeArrayModifierOffset,cfg.maxTreeArrayModifierOffset)


def buildingAugment(buildings):
    for building in buildings:
        bpy.context.view_layer.objects.active = building
        # random size (augment only width and height x,y)
        building.scale[0] = random.uniform(cfg.minBuildingScale,cfg.maxBuildingScale)
        building.scale[1] = random.uniform(cfg.minBuildingScale,cfg.maxBuildingScale)

        if (len(texture_list_building) > 0):
            mat = building.active_material
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]

            texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
            textureImageIndex = random.randint(0,len(texture_list_building)-1)
            texImage.image = bpy.data.images[texture_list_building[textureImageIndex]]

            mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
            if building.data.materials:
                building.data.materials[0] = mat
            else:
                building.data.materials.append(mat)


def origin_to_bottom(ob, matrix=Matrix()): # 
    me = ob.data
    mw = ob.matrix_world
    local_verts = [matrix @ Vector(v[:]) for v in ob.bound_box]
    o = sum(local_verts, Vector()) / 8
    o.y = min(v.y for v in local_verts)
    o = matrix.inverted() @ o
    me.transform(Matrix.Translation(-o))
    mw.translation = mw @ o

def reset_bike_rotation(obj):
    # for obj in rotated_bike_set:
        bpy.context.view_layer.objects.active = obj
        bpy.context.active_object.rotation_euler[0] = math.radians(0)
        bpy.context.active_object.rotation_euler[1] = math.radians(random.randint(-5,5))
        bpy.context.active_object.rotation_euler[2] = math.radians(random.randint(-15,15))
        bpy.context.active_object.location[2] = 0.000489


def organizeBikes():
    bikes = bpy.data.collections["bike"].objects 
    bikeList = []
    for bk in bikes:
        bk.hide_render = True
        bk.hide_viewport = True
        reset_bike_rotation(bk)
        bikeList.append(bk)
    numBike = random.randint(cfg.minBikes, cfg.maxBikes)  # select random number of bikes to render, hide the rest
    bikeSet = random.sample(bikeList, numBike)

    idxShuffled = random.sample(list(range(len(bikeSet))), len(bikeSet))

    maxBikeOffset = 45/numBike
    minBikeOffset = cfg.minBikeSpace
    minRequiredTotalBikeSpace = minBikeOffset * numBike
    maxRequiredTotalBikeSpace = maxBikeOffset * numBike
    avgTotal = (maxRequiredTotalBikeSpace + minRequiredTotalBikeSpace)/1.8
    currentX = random.uniform(-24.5, 23 - avgTotal)

    # print(currentX)
    for indx in idxShuffled:
        bikeSet[indx].hide_render = False
        bikeSet[indx].hide_viewport = False
        randOffsetX = random.uniform(minBikeOffset,maxBikeOffset)
        
        bikeSet[indx].location[0] = currentX+randOffsetX
        currentX = currentX+randOffsetX 

#        print(bikeSet[indx].name)
        if 'bikeShared' not in bikeSet[indx].name:
            randBikeColor = random.sample(['white','black','clr', 'other'],1)
            #change the color
            color = Color()
            hue = random.random() #* .1 # 0 - .2
            saturation = random.random() #* .7 # .2 - .8
            v = random.random() * .3
            if randBikeColor[0] == 'white':
                # print("white -----------")
                color.hsv = (0.0,0.0, 0.9)
            elif randBikeColor[0] == 'clr':
                color.hsv = (hue,saturation, v)
            else:
                # print("black -----------")
                color.hsv = (0.0,0.0, 0.0)
                
            
            rgba = [color.r, color.g, color.b, 1]
            bikeSet[indx].material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba
            
            #random metallic
            bikeSet[indx].material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.3,0.7)
            # random roughness
            bikeSet[indx].material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs[7].default_value = random.uniform(0.2,0.8)

    return bikeSet


def scene_cfg(camera, i):
    """configure the blender scene with random distributions."""
    annotationList = []

    obj_list = bpy.context.selectable_objects  # camera, objects
    ground_planes = []
    tree_objects = []
    building_objects = []

    # hide all objects
    for o in obj_list:
        if o.type == 'MESH':
            if 'ground' in o.name:
                ground_planes.append(o)
            elif 'tree' in o.name:
                tree_objects.append(o)
            elif 'building' in o.name:
                building_objects.append(o)

    
    cameraAugment()
    
    bikeSet = organizeBikes()

    # augment light
    lightAugment()

    # augment ground
    groundAugment(ground_planes)
    # augment a list of tree objects
    treeAugment(tree_objects)
    #augment bin
    yPathsBin = [-1,-2]
    augmentBin(yPathsBin)
    # # augment buildings
    buildingAugment(building_objects)

    yoloAnnotationList = []
    for j, obj in enumerate(bikeSet):
        obj.hide_render = False

        #random rotation z
        bpy.context.view_layer.objects.active = obj
        
        class_ = 0  # class label for object
        if random.choice(cfg.bikeRotationRatioZ):
            bpy.context.active_object.rotation_euler[2] = math.radians(random.choice([random.uniform(0,90),random.uniform(-90,0)]))
            class_ = 1
        if random.choice(cfg.bikeRotationRatioY):
            bpy.context.active_object.rotation_euler[1] = math.radians(random.choice([72,-75]))
            class_ = 2
            # locate the object up in z axis to avoid merging bike with the ground plane.
            bpy.context.active_object.location[2] = 0.228529 
        
        # update blender object world_matrices!
        bpy.context.view_layer.update()
        center = project_by_object_utils(camera, obj.location)  # object 2D center


        labels = [class_]
        labels.append(center[0])  # center x coordinate in image space
        labels.append(center[1])  # center y coordinate in image space
        corners = orderCorners(
            obj.bound_box)  # change order from blender to SSD paper

        kps = []
        # repeat = False
        for corner in corners:
            p = obj.matrix_world @ Vector(corner)  # object space to world space
            p = project_by_object_utils(camera, p)  # world space to image space
            labels.append(p[0])
            labels.append(p[1])
            if (p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1):
                v = 1  # v=1: labeled but not visible
            else:
                v = 2  # v=2: labeled and visible
            # 8 bounding box keypoints
            kps.append([p[0] * cfg.resolution_x, p[1] * cfg.resolution_y, v])

        # P=[RT] ground truth pose of the object in camera coordinates???
        P = camera.matrix_world.inverted() @ obj.matrix_world

        # compute bounding box either with 3D bbox or by going through vertices
        if (cfg.compute_bbox == 'tight'
            ):  # loop through all vertices and transform to image coordinates
            min_x, max_x, min_y, max_y = 1, 0, 1, 0
            vertices = obj.data.vertices
            for v in vertices:
                vec = project_by_object_utils(camera,
                                                obj.matrix_world @ Vector(v.co))
                x = vec[0]
                y = vec[1]
                if x > max_x:
                    max_x = x
                if x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                if y < min_y:
                    min_y = y
        else:  # use blenders 3D bbox (simple but fast)
            min_x = np.min([
                labels[3], labels[5], labels[7], labels[9], labels[11],
                labels[13], labels[15], labels[17]
            ])
            max_x = np.max([
                labels[3], labels[5], labels[7], labels[9], labels[11],
                labels[13], labels[15], labels[17]
            ])

            min_y = np.min([
                labels[4], labels[6], labels[8], labels[10], labels[12],
                labels[14], labels[16], labels[18]
            ])
            max_y = np.max([
                labels[4], labels[6], labels[8], labels[10], labels[12],
                labels[14], labels[16], labels[18]
            ])

        # save labels in txt file 
        x_range = max_x - min_x
        y_range = max_y - min_y
        labels.append(x_range)
        labels.append(y_range)

        # fix center point
        labels[1] = (max_x + min_x) / 2
        labels[2] = (max_y + min_y) / 2

        bikeRotation = []
        for ax in obj.rotation_euler:
                bikeRotation.append(ax)

        annotation = {
            "id": j,
            "image_id": i,
            "bbox": [
                min_x * cfg.resolution_x, min_y * cfg.resolution_y,
                x_range * cfg.resolution_x, y_range * cfg.resolution_y
            ],
            "category_id": class_,
            "rotation": bikeRotation,
        }
        annotationList.append(annotation)

        # Yolo style annotations:  [label x1 y1 x2 y2 ry rz]
        x = min_x * cfg.resolution_x
        y = min_y * cfg.resolution_y
        w = x_range * cfg.resolution_x
        h = y_range * cfg.resolution_y

        # Finding midpoints
        x_centre = (x + (x+w))/2
        y_centre = (y + (y+h))/2
        rx,ry,rz = bikeRotation
        if rz > 1.570795:
          rz = -0.5235987755982988
        if ry == 1.2566370964050293:
          ry = 1.570795
        elif ry == -1.3089969158172607:
          ry = -1.570795
        else: 
          ry = 0.0

        # Normalization
        x_centre = x_centre / cfg.resolution_x
        y_centre = y_centre / cfg.resolution_y
        w = w / cfg.resolution_x
        h = h / cfg.resolution_y
        # shift rotation annotations from [-1.570795:1.570795] to [0:3.14159] then normalize it to the range [0:1]
        ry = (ry + 1.570795)/ 3.14159  # (((+) + (-))/ (total) )
        rz = (rz + 1.570795)/ 3.14159  # (((+) + (-))/ (total) )

        # Limiting upto fix number of decimal places
        x_centre = format(x_centre, '.6f')
        y_centre = format(y_centre, '.6f')
        w = format(w, '.6f')
        h = format(h, '.6f')
        ry = format(ry, '.6f')
        rz = format(rz, '.6f')
            
        # Writing current object 
        yoloAnnotationList.append(str(class_) + ' ' + str(x_centre) +' ' +str(y_centre)+ ' '+ str(w)+ ' '+str(h)+ ' ' + str(ry) +' '+str(rz)+'\n')

    # COCO image annotation
    image = {
        "id": i,
        "file_name": "{:06}".format(i) + '.jpg',
        "height": cfg.resolution_y,
        "width": cfg.resolution_x,
    }

    # return image, annotation for COCO, yolo style annotations
    return image, annotationList, yoloAnnotationList

def render_cfg():
    """setup Blenders render engine (EEVEE or CYCLES) once"""
    # refresh the list of devices
    devices = bpy.context.preferences.addons["cycles"].preferences.get_devices()
    devices = devices[0]
    for d in devices:
        d["use"] = 1  # activate all devices
        print("activating device: " + str(d["name"]))
    if (cfg.use_cycles):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = cfg.samples
        bpy.context.scene.cycles.max_bounces = 8
        bpy.context.scene.cycles.use_denoising = cfg.use_cycles_denoising
        bpy.context.scene.cycles.use_adaptive_sampling = cfg.use_adaptive_sampling
        bpy.context.scene.cycles.adaptive_min_samples = 50
        bpy.context.scene.cycles.adaptive_threshold = 0.001
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Intel OpenImage AI denoiser on CPU
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = cfg.samples
    if (cfg.use_GPU):
        bpy.context.preferences.addons[
            'cycles'].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = 'GPU'

    # https://docs.blender.org/manual/en/latest/files/media/image_formats.html
    # set image width and height
    bpy.context.scene.render.resolution_x = cfg.resolution_x
    bpy.context.scene.render.resolution_y = cfg.resolution_y

    bpy.context.scene.render.image_settings.quality = cfg.renderQuality
    print("image quality:" + str(cfg.renderQuality))

def render(camera):
    """main loop to render images"""

    render_cfg()  # setup render config once
    annotations = []
    images = []

    start_time = datetime.datetime.now()

    #  render loop
    if (cfg.test):
        cfg.numberOfRenders = 1
    for i in range(cfg.numberOfRenders):

        bpy.context.scene.render.filepath = './DATASET/' + cfg.out_folder + '/images/{:06}.jpg'.format(i)
        image, annotation, yoloAnn = scene_cfg(camera, i)
        images.append(image)
        annotations.append(annotation)
        bpy.ops.render.render(write_still=True,
                              use_viewport=False)  # render current scene

        yoloTXT = open('./DATASET/' + cfg.out_folder + "/labels/{:06}.txt".format(i) , "w")
        yoloTXT.writelines(yoloAnn)
        yoloTXT.close()

        if i%20 == 0:
            saveCOCOlabel(images, annotations)  # save COCO annotation file at the end

    end_time = datetime.datetime.now()
    dt = end_time - start_time
    secondsPerRender = dt.seconds / cfg.numberOfRenders
    print('---------------')
    print('finished rendering')
    print('total render time (hh:mm:ss): ' + str(dt))
    print('average seconds per image: ' + str(secondsPerRender))

    return images, annotations    
    
    
def main():
    """
    call this script with 'blender --background {Bike.blend} -P main.py'

    edit the config.py file to change configuration parameters

    """
    if bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Load assets
    LoadData()

    camera = bpy.data.objects['Camera']

    Path('./DATASET/' + cfg.out_folder + '/labels').mkdir(parents=True, exist_ok=True)
    fileClasses = open('./DATASET/' + cfg.out_folder + "/classes.txt", "w")
    for cls in cfg.classes:
        fileClasses.write(cls + '\n')
    fileClasses.close()

    # images, annotations = render(camera, depth_file_output, Kdict)  # render loop
    images, annotations = render(camera) #, Kdict)  # render loop

    shutil.copy2('config.py',
                 'DATASET/' + cfg.out_folder)  # save config.py file
    shutil.copy2('main.py',
                 'DATASET/' + cfg.out_folder)  # save config.py file
    saveCOCOlabel(images, annotations)  # save COCO annotation file at the end

    return True


if __name__ == '__main__':
    main()

