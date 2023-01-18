"""
Created on xxx Sep 20 2022

@author: xxxx
"""


class cfg:
    def __init__(self):
        self.test = False # render one image 

        #  PATHS
        self.out_folder = 'SynthBPSet'  # render images will be saved to DATASET/out_folder
        self.object_texture_path = './assets/textures/ground' #'./environment'# './textures_realistic' #'./textures'
        self.grass_texture_path =  './assets/textures/grass'
        self.road_texture_path =  './assets/textures/road'
        self.building_texture_path =  './assets/textures/building'

        # AUGMENTATION
        self.minSidePavementScaleX = 130 # bike side pavement length
        self.maxSidePavementScaleX = 200 
        self.minGroundScaleX = 500   # walk pavement length
        self.maxGroundScaleX = 800  

        # trees
        self.minTreeArrayModifierOffset = 1.2  # min offset of the tree array modifier
        self.maxTreeArrayModifierOffset = 2.2
        self.minTreeSize = 2.0
        self.maxTreeSize = 9.0

        # buildings
        self.minBuildingScale = 25.0
        self.maxBuildingScale = 40.0

        # bins
        self.minBinSize = 0.9
        self.maxBinSize = 1.5

        # pedestrians
        self.minPlaneY = -16
        self.maxPlaneY = -14
        self.minPlaneX = -180
        self.maxPlaneX = 180
        self.minSizePerson = 0.98
        self.maxSizePerson = 1.01

        # Lights
        self.worldStringth_min = 0.1  
        self.worldStringth_max = 1.0  

        # Vertical camera path
        self.camPathEnableV = True #False
        self.camPathVa = -93
        self.camPathVb = -60
        # Horizontal camera path
        self.camPathEnableH = True #False
        self.camPathHa = -12 
        self.camPathHb = 12 

        # RENDERING CONFIG
        self.use_GPU = True
        self.use_cycles = True  # cycles or eevee
        self.use_cycles_denoising = False
        self.use_adaptive_sampling = False #True
        # self.resolution_x = 1920 #640  # pixel resolution
        # self.resolution_y = 1080 # 360
        self.resolution_x = 640  # pixel resolution
        self.resolution_y = 360
        self.samples = 512  # render engine samples
        self.renderQuality = 98 #%

        # Bikes
        self.minBikes = 7     
        self.maxBikes = 20
        self.minBikeSpace = 2.5 
        self.nextBikeStartX = 1.8 
        self.bikeRotationRatioY = [True,False,False,False]  #fallen
        self.bikeRotationRatioZ = [True,False]              #stand
        self.compute_bbox = 'tight'  # choose 'tight' or 'fast' (tight uses all vertices to compute a tight bbox but it is slower)

        # Classes
        self.classes = ['parked','rotated','fallen']

        # OUTPUT
        self.numberOfRenders = 3000  # how many rendered examples