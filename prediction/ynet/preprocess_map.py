
import os
import csv
import matplotlib.pyplot as plt
#%matplotlib inline
import json
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely import affinity
import cv2
from typing import List, Tuple
from functools import reduce
import pickle as pkl
import copy


Color = Tuple[float, float, float]

from functools import reduce

def geom_to_mask(layer_name, layer_geom, local_box, canvas_size):
    
    patch_x, patch_y, patch_h, patch_w = local_box
    patch_angle = 0.0

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]

    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for polygon in layer_geom:
        new_polygon = polygon.intersection(patch)
        if not new_polygon.is_empty:
            new_polygon = affinity.affine_transform(new_polygon,
                                                    [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_polygon = affinity.scale(new_polygon, xfact=scale_width, yfact=scale_height, origin=(0, 0))

            if new_polygon.geom_type is 'Polygon':
                new_polygon = MultiPolygon([new_polygon])
            # map_mask = self.mask_for_polygons(new_polygon, map_mask)

            if not new_polygon:
                return map_mask

            def int_coords(x):
                # function to round and convert to int
                return np.array(x)[:,:2].round().astype(np.int32)
            # print(type(new_polygon))
            try:
                exteriors = [int_coords(poly.exterior.coords) for poly in new_polygon]
                interiors = [int_coords(pi.coords) for poly in new_polygon for pi in poly.interiors]
                cv2.fillPoly(map_mask, exteriors, 1)
                cv2.fillPoly(map_mask, interiors, 0)
            except:
                pass
            
    # assert np.all(map_mask.shape[1:] == canvas_size)

    return map_mask
    
def get_map_mask(polygons, patchbox, angle_in_degrees, layer_names, canvas_size, color_dict):

    patch_x, patch_y, patch_h, patch_w = patchbox

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, angle_in_degrees, origin=(patch_x, patch_y), use_radians=False)

    map_geom = []
    colors = []
    for layer_name in layer_names:
        polygon_list = []
        try:
            for polygon in polygons[layer_name]:
                polygon = polygon.buffer(0)
                new_polygon = polygon.intersection(patch)
                
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -angle_in_degrees,
                                                    origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
        except:
            continue
        if polygon_list is None:
            continue
        map_geom.append((layer_name, polygon_list))
        if len(polygon_list)!=0:
            colors.append(color_dict[layer_name])
        

    local_box = (0.0, 0.0, patchbox[2], patchbox[3])
    map_mask = []
    for layer_name, layer_geom in map_geom:
        if len(layer_geom) == 0:
            continue
        layer_mask = geom_to_mask(layer_name, layer_geom, local_box, canvas_size)
        if layer_mask is not None:
            map_mask.append(np.array(layer_mask))
        
    masks = np.array(map_mask)

    images = []
    for mask, color in zip(masks, colors):
        images.append(change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color))
        
    image = combine(images)

    row_crop, col_crop = get_crops(250, 250, 250,
                                    250, 0.1,
                                    int(500 / 0.1)) #param5

    return image[row_crop, col_crop, :]

def get_crops(meters_ahead: float, meters_behind: float,
              meters_left: float, meters_right: float,
              resolution: float,
              image_side_length_pixels: int) -> Tuple[slice, slice]:
    """
    Crop the excess pixels and centers the agent at the (meters_ahead, meters_left)
    coordinate in the image.
    :param meters_ahead: Meters ahead of the agent.
    :param meters_behind: Meters behind of the agent.
    :param meters_left: Meters to the left of the agent.
    :param meters_right: Meters to the right of the agent.
    :param resolution: Resolution of image in pixels / meters.
    :param image_side_length_pixels: Length of the image in pixels.
    :return: Tuple of row and column slices to crop image.
    """

    row_crop = slice(0, int((meters_ahead + meters_behind) / resolution))
    col_crop = slice(int(image_side_length_pixels / 2 - (meters_left / resolution)),
                     int(image_side_length_pixels / 2 + (meters_right / resolution)))

    return row_crop, col_crop

def combine(data: List[np.ndarray]) -> np.ndarray:
    """
    Combine three channel images into a single image.
    :param data: List of images to combine.
    :return: Numpy array representing image (type 'uint8')
    """
    # All images in the dict are the same shape
    
    for i in range(len(data)):
        if data[0].shape != None:
            data_shape = data[0].shape
            break
    image_shape = data_shape

    base_image = np.zeros(image_shape).astype("uint8")
    return reduce(add_foreground_to_image, [base_image] + data)
    
def add_foreground_to_image(base_image: np.ndarray,
                            foreground_image: np.ndarray) -> np.ndarray:
    """
    Overlays a foreground image on top of a base image without mixing colors. Type uint8.
    :param base_image: Image that will be the background. Type uint8.
    :param foreground_image: Image that will be the foreground.
    :return: Image Numpy array of type uint8.
    """

    if not base_image.shape == foreground_image.shape:
        raise ValueError("base_image and foreground image must have the same shape."
                         " Received {} and {}".format(base_image.shape, foreground_image.shape))

    if not (base_image.dtype == "uint8" and foreground_image.dtype == "uint8"):
        raise ValueError("base_image and foreground image must be of type 'uint8'."
                         " Received {} and {}".format(base_image.dtype, foreground_image.dtype))

    img2gray = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(base_image, base_image, mask=mask_inv)
    img2_fg = cv2.bitwise_and(foreground_image, foreground_image, mask=mask)
    combined_image = cv2.add(img1_bg, img2_fg)
    return combined_image

def change_color_of_binary_mask(image: np.ndarray, color: Color) -> np.ndarray:
    """
    Changes color of binary mask. The image has values 0 or 1 but has three channels.
    :param image: Image with either 0 or 1 values and three channels.
    :param color: RGB color tuple.
    :return: Image with color changed (type uint8).
    """

    image = image * color

    # Return as type int so cv2 can manipulate it later.
    image = image.astype("uint8")

    return image

def list2Polygon(json_data):
    polygons_dict = {'Building':[], # 건물 -> 
            'Car_road1':[],
            'Car_road2':[],
            'Road':[],
            'Walkway':[],
            'Static_object':[],
            'Step':[], 
            'Walk_slope':[], 
            'Sharedway':[],
            'Cross_walk1':[], 
            'Cross_walk2':[], 
            'Slope':[],
            'Road_slope':[],
            }
    
    for type, polygons in json_data.items():
        for polygon in polygons:
            polygons_dict[type].append(Polygon(polygon))

    return polygons_dict



def check_indoor(file):
    indoor_list = ['hanyang', 'internat']
    for indoor in indoor_list:
        if indoor in file:
            return ['Road', 'Walkway', 'Static_object', 'Step', 'Building','Walk_slope']
    return ['Building', 'Car_road1', 'Walkway', 'Car_road2', 'Cross_walk1', 'Cross_walk2', 'Sharedway', 'Slope', 
                     'Road_slope',  ]


def int_coords(x):
    return np.array(x)[:2].round().astype(np.int32)

def makedir_recursive(path):
    
    if not os.path.exists(path):
        os.makedirs(path)

def get_image_crop(img, point_list, patchbox, angle_in_degrees, canvas_size,  W, H):
    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]

    patch_x, patch_y, patch_h, patch_w = patchbox
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w
    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0
    if len(point_list) == 1:
        point_list.append(point_list[0])
    polyline = LineString(point_list)
    polyline = affinity.rotate(polyline, angle_in_degrees,
                                    origin=(patch_x, patch_y), use_radians=False)
    polyline = affinity.affine_transform(polyline,
                                            [0.0, 1.0, 1.0, 0.0, trans_x, trans_y])
                                            # [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
    polyline = affinity.scale(polyline, xfact=scale_width, yfact=scale_height, origin=(0, 0))
    polyline = np.array([int_coords(poly) for poly in polyline.coords])

    cur_x = polyline[10, 0]
    cur_y = polyline[10, 1]
    x_l = cur_x - W//2
    x_u = cur_x + W//2
    y_l = cur_y - H//2
    y_u = cur_y + H//2
    return img[x_l:x_u, y_l:y_u]

def change_keys(json_data):
    change_key_dict = {
            'building':'Building',
            'car_road1':'Car_road1',
            'car_road2':'Car_road1',
            'Intersection':'Car_road1',
            'road':'Car_road1',
            'Road':'Car_road1',
            'walkway':'Walkway',
            'walkaway':'Walkway',
            'merge':'Static_object',
            'Merge':'Static_object',
            'free space':'Walkway',
            'Free space':'Walkway',
            'keep out':'Walkway',
            'Keep out':'Walkway',
            'sharedway':'Sharedway',
            'Crossing':'Sharedway',
            'cross_walk1':'Sharedway',
            'cross_walk2':'Sharedway',
            'U turn':'Cross_walk2',
            'Island':'Car_road1'}
    combined_list = [item for item in change_key_dict.keys()] + [item for item in change_key_dict.values()]
    del_list = []
    for k, v in json_data.items():
        if k not in combined_list:
            del_list.append(k)
    for del_k in del_list:
        del json_data[k]
            
    for k, v in change_key_dict.items():
        try:
            if v in json_data.keys():
                json_data[v] += json_data[k]
            else:
                json_data[v] = copy.deepcopy(json_data[k])

            del json_data[k]
        except:
            continue
    return json_data





def scene_patch_canvas_size(scene):
    patch_canvas = {   #+Left  +Down   +Scaledown +Resolution +Clockwise
            'Cafeteria'                 :(-20,  0,      65,     65/400*5000,      0),
            'Corridor'                  :(15,   0,      135,    135/400*5000,     0),
            'Lobby'                     :(-10,  12,     110,    110/400*5000,     0),
            'Hallway'                   :(15,   0,      65,     65/400*5000,      0),
            'Courtyard'                 :(10,   0,      75,     75/400*5000,      0),
            'Subway_Entrance'           :(60,   -20,    250,    250/400*5000,     0),
            'Three_way_Intersection'    :(30,   45,     270,    270/400*5000,     0),
            'Crossroad'                 :(40,   0,      300,    300/400*5000,     0),
            'Outdoor_Alley'             :(40,   0,      300,    300/400*5000,     0),
            'Cafe_Street'               :(-10,  -70,    320,    320/400*5000,     0),
        }
    
    canvas_size = int(patch_canvas[scene][3])
    canvas_size = int(patch_canvas[scene][2]*50/4)
    angle_in_degrees = patch_canvas[scene][4]
    patchbox = (patch_canvas[scene][0], patch_canvas[scene][1], patch_canvas[scene][2], patch_canvas[scene][2])
    return patchbox, (canvas_size, canvas_size), angle_in_degrees




color_dict = {
        'Building':[5 ,5, 5],
        'Car_road1':[15, 15, 15],
        'Car_road2':[25, 25, 25],
        'Road':[35, 35, 35],
        'Walkway':[45, 45, 45],
        'Static_object':[55, 55, 55],
        'Step':[65, 65, 65], 
        'Walk_slope':[75, 75, 75], 
        'Slope':[85, 85, 85],
        'Sharedway':[95, 95, 95],
        'Cross_walk1':[105, 105, 105], 
        'Cross_walk2':[115, 115, 115], 
        'Road_slope':[125, 125, 125],
        }


layer_names = ['Building', 'Car_road1', 'Walkway', 'Sharedway', 'Static_object', ]


   
scene_list = ['Cafeteria', 'Corridor', 'Lobby', 'Hallway', 'Courtyard', 'Subway_Entrance', 'Three_way_Intersection', 'Crossroad', 'Cafe_Street', 'Outdoor_Alley', ]
json_dir = './data/semantic_maps_json/'
save_dir = './data/semantic_map'

makedir_recursive(save_dir)
             
for scene in scene_list:
    
    json_file = os.path.join(json_dir, f'{scene}.json')
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    patchbox, canvas_size, angle_in_degrees = scene_patch_canvas_size(scene)

    json_data = change_keys(json_data)
    polygons = list2Polygon(json_data)
    raster_img = get_map_mask(polygons, patchbox, angle_in_degrees, layer_names, canvas_size, color_dict)
    cv2.imwrite(f'{save_dir}/{scene}.jpg', raster_img)
    