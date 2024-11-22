"""
Interface for the TITAN dataset:

Ruthvik.

"""
import cv2
import csv
import sys
import pickle
import random
#import imageio
import pandas as pd
import numpy as np
# import seaborn as sns
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from os import makedirs, listdir,environ
from collections import Counter
from sklearn.model_selection import train_test_split, KFold
from os.path import join, abspath, isfile, isdir,basename, normpath,exists


# Set a global theme for Seaborn
# sns.set_theme(style="whitegrid")

class TITAN():
    def __init__(self, regen_database=False, data_path=''):
        """
        TITAN dataset Class constructor
        :param regen_database: Whether generate the database or not
        :param data_path: The path to the root directory of the dataset.
        """
        self._name = 'titan'
        self._image_ext = '.png'
        self._regen_database = regen_database

        # Paths
        self._titan_path = data_path if data_path else self._get_default_path()
        assert isdir(self._titan_path), \
            'The provided path {} does not exist. Kindly recheck the path provided'.format(self._titan_path)

        self._annotation_path = join(self._titan_path,'annotations')
        self._annotation_vehicle_path = join(self._titan_path,'annotations_vehicle')
        self._images_path = join(self._titan_path,'images_anonymized')

    # Path generators
    @property
    def cache_path(self):
        """
        Generates a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._titan_path, 'data_cache'))
        if not isdir(cache_path):
            makedirs(cache_path,exist_ok=True)
        return cache_path

    def _get_default_path(self):
        """
        Returns the default path where the titan dataset is expected to be installed.
        """
        return join('data','titan_data','dataset') # ./titan_data/dataset
    
    def _get_splits_path(self):
        return join(self._titan_path,'splits')

    def _get_image_set_ids(self, image_set):
        """
        Returns default image clip ids
        :param image_set: Image clip split
        :return: clip ids of the image set
        """
        root_split_path = self._get_splits_path()
        file_name = join(root_split_path,'{}_set.txt'.format(image_set))
        with open(file_name,'r') as f:
            clips_list = f.readlines()
        clips_list = [c.strip() for c in clips_list] # Strip the '\n' character'
        return clips_list

    def _get_image_path(self, cid, fid):
        """
        Generates and returns the image path given ids
        :param sid: Clip id
        :param fid: Frame id
        :return: Return the path to the given image
        """
        return join(self._images_path, cid, 'images',
                    '{:06d}.png'.format(fid)) # padded with 0s to reach len 6
    
    # def tlhw_to_xyxy(self,top, left, height, width):
    #     """
    #     Function to convert from the top,left,height,width format provided by TITAN
    #     to the x1y1x2y2 format similar to PIE, JAAD, IDDP.
    #     """
    #     x1 = left
    #     y1 = top
    #     x2 = left + width
    #     y2 = top + height
    #     return x1, y1, x2, y2

    # Visual helpers
    def update_progress(self, progress):
        """
        Creates a progress bar
        :param progress: The progress thus far
        """
        barLength = 20  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)

        block = int(round(barLength * progress))
        text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def _print_dict(self, dic):
        """
        Prints a dictionary, one key-value pair per line
        :param dic: Dictionary
        """
        for k, v in dic.items():
            print('%s: %s' % (str(k), str(v)))

    def process_clip_csv(self,file_path):
        df = pd.read_csv(file_path)
        person_frames = df[df['label'] == 'person']['frames'].unique()
        num_unique_frames = len(person_frames)
        non_padded_frames = [int(frame.split('.')[0]) for frame in person_frames]
        return num_unique_frames, non_padded_frames

    def create_summary_csv(self):
        print("Generating a summary of the annotated frames across the clips.")
        directory = self._annotation_path
        summary_data = []

        for file_name in listdir(directory):
            if file_name.endswith('.csv') and file_name.startswith('clip_'):
                clip_id = int(file_name.split('_')[1].split('.')[0])
                file_path = join(directory, file_name)
                num_unique_frames, non_padded_frames = self.process_clip_csv(file_path)
                row = [clip_id, num_unique_frames] + non_padded_frames
                summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(join(directory, 'annotated_frames.csv'), header=False, index=False)

    # def process_xml_files(self,directory):
    #     """
    #     Processes the csv files in the given directory and generates a csv file of annotated frames
    #     :param directory: The directory containing the csv files
    #     """
    #     xml_files = [f for f in listdir(directory) if f.endswith('.xml')]
    #     set_name = basename(normpath(directory))
    #     print("Processing set", set_name)

    #     if exists(join(directory,'{}_annotated_frames.csv'.format(set_name))):
    #         raise Exception('{}_annotated_frames.csv already exists. Delete the file and try again.'.format(set_name))
        
    #     with open(join(directory,'{}_annotated_frames.csv'.format(set_name)), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)

    #         for xml_file in xml_files:
    #             xml_path = join(directory, xml_file)
    #             tree = ET.parse(xml_path)
    #             root = tree.getroot()

    #             name = root.find('meta/task/name').text

    #             frames = set()
    #             for track in root.findall('track'):
    #                 if track.get('label') != 'pedestrian':
    #                     continue
    #                 for box in track.findall('box'):
    #                     frame = int(box.get('frame'))
    #                     outside = box.get('outside')
    #                     if outside == '0':
    #                         frames.add(frame)

    #             frames = sorted(frames)
    #             row = [name, len(frames)] + frames
    #             writer.writerow(row)


    
    def calculate_transformed_bbox(self, original_crop_size, target_size):
        """
        Calculate the transformed bounding box coordinates in a resized and padded square image,
        assuming the top-left corner of the cropped image as (0,0).

        Args:
        original_crop_size: Size of the cropped region (height, width).
        target_size: The size of the square image after resizing and padding (height and width are the same).

        Returns:
        Transformed bounding box coordinates.
        """
        crop_height, crop_width = original_crop_size

        # Determine the new width and height based on aspect ratio
        if crop_height > crop_width:
            new_height = target_size
            new_width = int(crop_width * (target_size / crop_height))
        else:
            new_width = target_size
            new_height = int(crop_height * (target_size / crop_width))

        # Calculate padding
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2

        # Transformed coordinates
        # The top-left corner (x1, y1) is (pad_x, pad_y)
        # The bottom-right corner (x2, y2) is (pad_x + new_width, pad_y + new_height)
        transformed_x1 = pad_x
        transformed_y1 = pad_y
        transformed_x2 = pad_x + new_width
        transformed_y2 = pad_y + new_height

        return [transformed_x1, transformed_y1, transformed_x2, transformed_y2]

    # Annotation processing helpers
    def _map_text_to_scalar(self, label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'occlusion': {'None': 0, 'Part': 1, 'Full': 2},
                   'CrossingBehavior': {'CU': 0, 'CFU': 1, 'CD': 2, 'CFD': 3, 'N/A': 4, 'CI': -1},
                   'TrafficInteraction': {'WTT': 0, 'HG': 1, 'Other': 2, 'N/A': 3},
                   'PedestrianActivity': {'Walking': 0, 'MS': 1, 'N/A': 2},
                   'AttentionIndicators': {'LOS': 0, 'FTT': 1, 'NL': 2, 'DB': 3},
                   'SocialDynamics': {'GS': 0, 'CFA': 1, 'AWC': 2, 'N/A': 3},
                   'StationaryBehavior': {'Sitting': 0, 'Standing': 1, 'IWA': 2, 'Other': 3, 'N/A': 4},
                   'crossing': {'no': 0, 'yes': 1}, # whether the person is seen crossing in front of the ego-vehicle. Literally traversing the width of the ego-vehicle.
                   'age': {'child': 0, 'teenager': 1, 'adult': 2, 'senior': 3},
                   'carrying_object': {'none': 0, 'small': 1, 'large': 2},
                   'crossing_motive': {'yes': 1, 'maybe': 0.5, 'no': 0}, # changed from 0,1,2 to 1, 0.5, 0
                   'crosswalk_usage': {'yes': 0, 'no': 1, 'partial': 2, 'N/A': 3},
                   'intersection_type': {'NI': 0, 'U-turn': 1, 'T-right': 2, 'T-left': 3, 'four-way': 4, 'Y-intersection': 5},
                   'motion_direction': {'OW': 0, 'TW': 1},
                   'signalized_type': {'N/A': 0, 'C': 1, 'S': 2, 'CS': 3},
                   'road_type': {'main': 0, 'secondary': 1, 'street': 2, 'lane': 3},
                   'location_type': {'urban': 0, 'rural': 1, 'commercial': 2, 'residential': 3},
                   'vehicle': {'car': 0, 'motorcycle': 1, 'bicycle': 2, 'auto': 3, 'bus': 4, 'cart': 5, 'truck': 6, 'other': 7},
                   'traffic_light': {'pedestrian': 0, 'vehicle': 1},
                   'state': {'red': 0, 'orange': 1, 'green': 2}}

        return map_dic[label_type][value]

    def _map_scalar_to_text(self, label_type, value):
        """
        Maps a scalar value to a text label. Reverse mapping of _map_text_to_scalar.
        :param label_type: The label type
        :param value: The scalar to be mapped
        :return: The text label
        """
        map_dic = {'occlusion': {0: 'None', 1: 'Part', 2: 'Full'},
                   'CrossingBehavior': {0: 'CU', 1: 'CFU', 2: 'CD', 3: 'CFD', 4: 'N/A', -1: 'CI'},
                   'TrafficInteraction': {0: 'WTT', 1: 'HG', 2: 'Other', 3: 'N/A'},
                   'PedestrianActivity': {0: 'Walking', 1: 'MS', 2: 'N/A'},
                   'AttentionIndicators': {0: 'LOS', 1: 'FTT', 2: 'NL', 3: 'DB'},
                   'SocialDynamics': {0: 'GS', 1: 'CFA', 2: 'AWC', 3: 'N/A'},
                   'StationaryBehavior': {0: 'Sitting', 1: 'Standing', 2: 'IWA', 3: 'Other', 4: 'N/A'},
                   'crossing': {0: 'no', 1: 'yes'},
                   'age': {0: 'child', 1: 'teenager', 2: 'adult', 3: 'senior'},
                   'carrying_object': {0: 'none', 1: 'small', 2: 'large'},
                   'crossing_motive': {1: 'yes', 0.5: 'maybe', 0: 'no'}, # changed from 0,1,2 to 1, 0.5, 0
                   'crosswalk_usage': {0: 'yes', 1: 'no', 2: 'partial', 3: 'N/A'},
                   'intersection_type': {0: 'NI', 1: 'U-turn', 2: 'T-right', 3: 'T-left', 4: 'four-way', 5: 'Y-intersection'},
                   'motion_direction': {0: 'OW', 1: 'TW'},
                   'signalized_type': {0: 'N/A', 1: 'C', 2: 'S', 3: 'CS'},
                   'road_type': {0: 'main', 1: 'secondary', 2: 'street', 3: 'lane'},
                   'location_type': {0: 'urban', 1: 'rural', 2: 'commercial', 3: 'residential'},
                   'vehicle': {0: 'car', 1: 'motorcycle', 2: 'bicycle', 3: 'auto', 4: 'bus', 5: 'cart', 6: 'truck', 7: 'other'},
                   'traffic_light': {0: 'pedestrian', 1: 'vehicle'},
                   'state': {0: 'red', 1: 'orange', 2: 'green'}}

        return map_dic[label_type][value]

    def _get_annotations(self, cid):
        """
        Generates a dictionary of annotations by going through the csv file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of annotations
        """
        path_to_file = join(self._annotation_path,cid +'.xml')

        tree = ET.parse(path_to_file)
        ped_annt = 'pedestrian_annotations'
        traffic_annt = 'traffic_annotations'

        annotations = {}
        annotations['num_frames'] = int(tree.find("./meta/task/size").text)
        annotations['width'] = int(tree.find("./meta/task/original_size/width").text)
        annotations['height'] = int(tree.find("./meta/task/original_size/height").text)
        annotations[ped_annt] = {}
        annotations[traffic_annt] = {}
        tracks = tree.findall('./track')
        for t in tracks:
            boxes = t.findall('./box')
            obj_label = t.get('label')

            # NOOTE - POI annotations are not tracks. But some annotators seem to have done that mistake. So while you fix that,
            # you can ignore the POI annotations.
            if obj_label == "POI":
                continue

            obj_id = boxes[0].find('./attribute[@name=\"id\"]').text
            if obj_id == None:
                print('object id none lololP{}'.format(obj_id),setid,vid,boxes[0].get('frame'))
                print("Skipping")
                continue

            if obj_label == 'pedestrian':
                beh_mapper = {'CrossingBehavior': 'CB', 'TrafficInteraction': 'TI', 'PedestrianActivity': 'PA',
                              'AttentionIndicators': 'AI','SocialDynamics': 'SD','StationaryBehavior': 'SB'}
                annotations[ped_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': []}
                annotations[ped_annt][obj_id]['behavior'] = {'CrossingBehavior': [], 'TrafficInteraction': [], 'PedestrianActivity': [], 'AttentionIndicators': [],
                                                             'SocialDynamics': [], 'StationaryBehavior': []}
                for b in boxes:
                    # Exclude the annotations that are outside of the frame. Recall how we haven't extracted the frames which are outside the track either.
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[ped_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    occ = self._map_text_to_scalar('occlusion', b.find('./attribute[@name=\"occlusion\"]').text)
                    annotations[ped_annt][obj_id]['occlusion'].append(occ)
                    annotations[ped_annt][obj_id]['frames'].append(int(b.get('frame')))
                    for beh in annotations[ped_annt][obj_id]['behavior']:
                        # Read behavior tags for each frame and add to the database
                        annotations[ped_annt][obj_id]['behavior'][beh].append(
                            self._map_text_to_scalar(beh, b.find('./attribute[@name=\"' + beh_mapper[beh] + '\"]').text))

            else: # for other object types - namely traffic_light, vehicle, bus_station, crosswalk.
                obj_type = boxes[0].find('./attribute[@name=\"type\"]')
                if obj_type is not None:
                    obj_type = self._map_text_to_scalar(obj_label,
                                                        boxes[0].find('./attribute[@name=\"type\"]').text)

                annotations[traffic_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': [],
                                                     'obj_class': obj_label,
                                                     'obj_type': obj_type,
                                                     'state': []}

                for b in boxes:
                    # Exclude the annotations that are outside of the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[traffic_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    annotations[traffic_annt][obj_id]['occlusion'].append(int(b.get('occluded')))
                    annotations[traffic_annt][obj_id]['frames'].append(int(b.get('frame')))
                    if obj_label == 'traffic_light':
                        annotations[traffic_annt][obj_id]['state'].append(self._map_text_to_scalar('state',
                                                                                                    b.find(
                                                                                                        './attribute[@name=\"state\"]').text))
        return annotations

    def _get_ped_attributes(self, setid, vid):
        """
        Generates a dictionary of attributes by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of attributes
        """
        cam = setid.split('_')[0]
        if cam == "gp":
            cam = "gopro"
        elif cam == "d":
            cam = "ddpai"
        path_to_file = join(self._annotation_path,cam, setid, vid +'.xml')

        tree = ET.parse(path_to_file)

        tracks = tree.findall('./track')
        attributes = {}
        for t in tracks:
            boxes = t.findall('./box')
            obj_label = t.get('label')

            # NOOTE - POI annotations are not tracks. But some annotators seem to have done that mistake. So while you fix that,
            # you can ignore the POI annotations.
            if obj_label != "POI":
                continue

            ped_id = boxes[0].find('./attribute[@name=\"id\"]').text
            attributes[ped_id] = {}
            for attribute in boxes[0].findall('./attribute'):
                k = attribute.get('name')
                v = attribute.text
                if 'id' in k:
                    continue
                try:
                    attributes[ped_id][k] = int(v) # For the attributes that are integers, namely, group_size and crossing_point.
                except ValueError:
                    attributes[ped_id][k] = self._map_text_to_scalar(k, v)

        return attributes

    def _get_vehicle_attributes(self, cid):
        """
        Generates a dictionary of vehicle attributes by parsing the video XML file
        :param cid: The clip id to get the vehicle values for.
        :return: A dictionary of vehicle attributes (obd sensor recording)
        """
        path_to_file = join(self._annotation_vehicle_path, cid, 'synced_sensors.csv')

        veh_attributes = {}

        if not exists(path_to_file):
            print("File not found: {}".format(path_to_file))
            return veh_attributes

        with open(path_to_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                image_path = parts[1]
                frame_id = int(image_path.split('/')[-1].split('.')[0])
                obd_speed = float(parts[3])
                ang_vel = float(parts[5])

                veh_attributes[frame_id] = {
                    'OBD_speed': obd_speed,
                    'ang_vel': ang_vel
                }

        return veh_attributes
    
    def process_clip(self,file_path, global_id_counter, global_pedestrian_id_map):
        df = pd.read_csv(file_path)
        clip_id = basename(file_path).split('.')[0]
        print("Processing annotations for clip id",clip_id)

        # Get unique frames and their count
        unique_frames = df['frames'].unique()
        num_frames = len(unique_frames)

        # Get image dimensions (assuming all images in a clip have the same dimensions)
        sample_frame = unique_frames[0].split('.')[0]
        sample_image_path = self._get_image_path(clip_id, int(sample_frame))
        with Image.open(sample_image_path) as img:
            width, height = img.size

        # Dictionary to store pedestrian annotations
        pedestrian_annotations = {}
        for _, row in df.iterrows():
            if row['label'] == 'person':
                obj_track_id = row['obj_track_id']
                if (clip_id, obj_track_id) not in global_pedestrian_id_map:
                    # If it's a new object instance
                    global_pedestrian_id = "t_{}".format(global_id_counter)
                    global_pedestrian_id_map[(clip_id, obj_track_id)] = global_pedestrian_id
                    global_id_counter += 1
                else:
                    # If it's an existing object instance.
                    global_pedestrian_id = global_pedestrian_id_map[(clip_id, obj_track_id)]
                
                frame_number = int(row['frames'].split('.')[0])
                # bbox = [row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height']]
                bbox = [float(row['left']), float(row['top']), float(row['left'] + row['width']), float(row['top'] + row['height'])]

                if global_pedestrian_id not in pedestrian_annotations:
                    pedestrian_annotations[global_pedestrian_id] = {
                        'frames': [],
                        'bbox': []
                    }

                pedestrian_annotations[global_pedestrian_id]['frames'].append(frame_number)
                pedestrian_annotations[global_pedestrian_id]['bbox'].append(bbox)

        # Compile the clip data
        clip_data = {
            'num_frames': num_frames,
            'width': width,
            'height': height,
            'pedestrian_annotations': pedestrian_annotations
        }

        return global_id_counter, clip_id, clip_data

    def create_clip_dictionaries(self,directory):
        global_id_counter = 1
        global_pedestrian_id_map = {}
        clips_data = {}

        for file_name in listdir(directory):
            if file_name.endswith('.csv') and file_name.startswith('clip_'):
                file_path = join(directory, file_name)
                global_id_counter, clip_id, clip_data = self.process_clip(file_path, global_id_counter, global_pedestrian_id_map)
                clips_data[clip_id] = clip_data

        print("Total number of pedestrians in the dataset:",global_id_counter)
        self.clips_data = clips_data
        return clips_data

    def get_clip_data(self, clip_id):
        return self.clips_data.get(clip_id, "Clip ID not found")


    def generate_database(self):
        """
        Generates and saves a "database" of the titan dataset by integrating all annotations. Basically it's just a *dictionary* 
        containing all annotations in the dataset.

        Dictionary structure:
            'clip_id'(str): {
                'num_frames': int
                'width': int
                'height': int
                'traffic_annotations'(str): {
                    'obj_id'(str): {
                        'frames': list(int)
                        'occlusion': list(int)
                        'bbox': list([x1, y1, x2, y2]) (float)
                        'obj_class': str,
                        'obj_type': str,    # only for traffic lights, vehicles.
                        'state': list(int)  # only for traffic lights - red, orange or green
                'pedestrian_annotations'(str): {
                    'pedestrian_id'(str): {
                        'frames': list(int)
                        'occlusion': list(int)
                        'bbox': list([x1, y1, x2, y2]) (float)
                        'behavior'(str): {
                            'CrossingBehavior': list(int)
                            'TrafficInteration': list(int)
                            'PedestrianActivity': list(int)
                            'AttentionIndicators': list(int)
                            'SocialDynamics': list(int)
                            'StationaryBehavior': list(int)
                        'attributes'(str): {
                             'age': int
                             'carrying_object': int
                             'id': str
                             'crossing_motive': int
                             'crossing': int
                             'location_type': int
                             'crossing_point': int
                             'crosswalk_usage': int
                             'intersection_type': int
                             'signalized_type': int
                             'road_type': int
                             'group_size': int
                             'motion_direction': int
                'vehicle_annotations'(str){
                    'frame_id'(int){
                          'accT': float
                          'accX': float
                          'accY': float
                          'accZ': float
                          'OBD_speed': float

        :return: A database dictionary
        """

        print('---------------------------------------------------------')
        print("Generating database for the TITAN dataset...")

        cache_file = join(self.cache_path, 'titan_database.pkl')
        if isfile(cache_file) and not self._regen_database:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('TITAN annotations loaded from {}'.format(cache_file))
            return database

        print("####################################################################################")
        print("Database cache file not found. Generating database for the TITAN dataset...")
        print("####################################################################################")
        
        # Path to the folder annotations
        clip_ids = []
        clip_ids.extend([f.split('.')[0] for f in listdir(self._annotation_path)])
        clip_ids.remove('annotated_frames')

        print("The number of clips found in the dataset are:", len(clip_ids))

        # Read the content of set folders
        clips_data = self.create_clip_dictionaries(self._annotation_path)
        database = {}
        for cid in clip_ids:
            database[cid] = {}
            database[cid] = clips_data[cid]
            # Commenting out the below temporarily, TODO later
            # vid_attributes = self._get_ped_attributes(cid)
            database[cid]['vehicle_annotations'] = self._get_vehicle_attributes(cid)
            # for ped in database[cid]['pedestrian_annotations']:
            #     database[cid]['pedestrian_annotations'][ped]['attributes'] = vid_attributes[ped]

        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database

    # Process pedestrian ids
    def _get_pedestrian_ids(self):
        """
        Returns all pedestrian ids
        :return: A list of pedestrian ids
        """
        annotations = self.generate_database()
        pids = []
        for cid in sorted(annotations):
            pids.extend(annotations[cid]['pedestrian_annotations'].keys())
        return pids


    def _get_category_ids(self,image_set):
        """
        The image set is of form test_{category}, where category is a category a pedestrian in the test
        set belong to.
        :return: Return the pedestrian ids belonging to this category.
        """
        category = image_set.split('_')[1]
        if category == 'NA':
            category = 'N/A' # Consistent with what's there in the annotations.
        # 'signalized': {'N/A': 0, 'C': 1, 'S': 2, 'CS': 3}
        set_ids = self._get_image_set_ids(image_set.split('_')[0])
        annotations = self.generate_database()
        pids = []
        for sid in set_ids:
            print("Going through set",sid,"and adding only those pids")
            for vid in annotations[sid].keys():
                for pid in annotations[sid][vid]['pedestrian_annotations'].keys():
                    try:
                        value = annotations[sid][vid]['pedestrian_annotations'][pid]['attributes']['signalized_type']
                    except:
                        # pid doesn't have corresponding attributes
                        continue
                    if value == self._map_text_to_scalar('signalized_type', category):
                        pids.append(pid)
        return pids

    def _get_crosscategory_ids(self,image_set):
        """
        The image set is of form test_{category}, where category is a category a pedestrian in the test
        set belong to.
        :return: Return the pedestrian ids belonging to this category.
        """
        # Get the set_ids for the data split.
        set_ids = self._get_image_set_ids(image_set.split('_')[0])
        crossing_scenario = image_set.split('_')[1]
        annotations = self.generate_database()
        pids = []
        for sid in set_ids:
            print("Going through set",sid)
            for vid in annotations[sid].keys():
                for pid in annotations[sid][vid]['pedestrian_annotations'].keys():
                    crossing_behavior = annotations[sid][vid]['pedestrian_annotations'][pid]['behavior']['CrossingBehavior']
                    if crossing_scenario == 'UD':
                        if (np.array(crossing_behavior) == 0).sum() + (np.array(crossing_behavior) == 1).sum() > 5:
                            pids.append(pid)
                    elif crossing_scenario == 'D':
                        if (np.array(crossing_behavior) == 2).sum() + (np.array(crossing_behavior) == 3).sum() > 5:
                            pids.append(pid)
        return pids


    # Trajectory data generation
    def _get_data_ids(self, image_set, params):
        """
        Generates set ids and ped ids (if needed) for processing
        :param image_set: Image-set to generate data
        :param params: Data generation params
        :return: Set and pedestrian ids
        """
        _pids = None
        if params['data_split_type'] == 'default':
            set_ids = self._get_image_set_ids(image_set)
        else:
            set_ids = self._get_image_set_ids('all')
        # Though we return all set_ids above in case data_split_type is not default, the pids are gotten
        # from (atleast in my get_category_ids function) the corresponding set (in this case 'test' only.)
        if params['data_split_type'] == 'random':
            _pids = self._get_random_pedestrian_ids(image_set, **params['random_params'])
        elif params['data_split_type'] == 'kfold':
            _pids = self._get_kfold_pedestrian_ids(image_set, **params['kfold_params'])
        elif params['data_split_type'] == 'test':
            # print("Entering this phase")
            _pids = self._get_category_ids(image_set)
        elif params['data_split_type'] == 'crosstest':
            _pids = self._get_crosscategory_ids(image_set)

        return set_ids, _pids

    def _squarify(self, bbox, ratio, img_width):
        """
        Changes the ratio of bounding boxes to a fixed ratio
        :param bbox: Bounding box
        :param ratio: Ratio to be changed to
        :param img_width: Image width
        :return: Squarified boduning box
        """
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * ratio - width

        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2

        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    def _height_check(self, height_rng, frame_ids, boxes, images):
        """
        Checks whether the bounding boxes are within a given height limit. If not, the data where
        the boxes aren't in the height range are ignored while creating the sequence.
        :param height_rng: Height limit [lower, higher]
        :param frame_ids: List of frame ids
        :param boxes: List of bounding boxes
        :param images: List of images
        :param occlusion: List of occlusions
        :return: The adjusted data sequences
        """
        imgs, box, frames = [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)
                imgs.append(images[i])
                frames.append(frame_ids[i])
        return imgs, box, frames

    

    def _get_center(self, box):
        """
        Calculates the center coordinate of a bounding box
        :param box: Bounding box coordinates
        :return: The center coordinate
        """
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def generate_data_trajectory_sequence(self, image_set, **opts):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param opts:
                'fstride': Frequency of sampling from the data.
                'height_rng': The height range of pedestrians to use.
                'squarify_ratio': The width/height ratio of bounding boxes. A value between (0,1]. 0 the original
                                        ratio is used.
                'data_split_type': How to split the data. Options: 'default', predefined sets, 'random', randomly split the data,
                                        and 'kfold', k-fold data split (NOTE: only train/test splits).
                'seq_type': Sequence type to generate. Options: 'trajectory', generates tracks, 'crossing', generates
                                  tracks up to 'crossing_point', 'intention' generates tracks similar to human experiments
                'seq_end': The end of the sequence. Options: 'crossing_point', 'track_end'
                'min_track_size': Min track length allowable.
                'random_params: Parameters for random data split generation. (see _get_random_pedestrian_ids)
                'kfold_params: Parameters for kfold split generation. (see _get_kfold_pedestrian_ids)
        :return: Sequence data
        """
        params = {'fstride': 1,
                  'sample_type': 'all',
                  'height_rng': [0, float('inf')],
                  'squarify_ratio': 0,
                  'data_split_type': 'default',  # kfold, random, default
                  'seq_type': 'crossing',
                  'train_seq_end': 'crossing_point', # crossing_point, track_end
                  'test_seq_end': 'crossing_point', # crossing_point, track_end
                  'val_seq_end': 'crossing_point',  # crossing_point, track_end
                  'min_track_size': 16, # 16 frame observation period as outlined in the evaluation protocol for crossing.
                  'random_params': {'ratios': None,
                                    'val_data': True,
                                    'regen_data': False},
                  'kfold_params': {'num_folds': 5, 'fold': 1}}

        # Overwrite params from the input, if any.
        for i in opts.keys():
            params[i] = opts[i]

        print('---------------------------------------------------------')
        print("Generating trajectory sequence data for sequence type: %s" % params['seq_type'])
        self._print_dict(params)
        annot_database = self.generate_database()
        if params['seq_type'] == 'trajectory':
            sequence_data = self._get_trajectories(image_set, annot_database, **params)
        elif params['seq_type'] == 'crossing':
            sequence_data = self._get_crossing(image_set, annot_database, **params)
        elif params['seq_type'] == 'intention':
            sequence_data = self._get_intention(image_set, annot_database, **params)
        elif params['seq_type'] == 'post_crossing':
            sequence_data = self._get_post_crossing(image_set, annot_database, **params)
        return sequence_data

    def _get_trajectories(self, image_set, annotations, **params):
        """
        Generates trajectory data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """
        print('---------------------------------------------------------')
        print("Generating trajectory data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq = [], []
        obds_seq, angv_seq= [], []

        # For testing trajectory prediction for certain scenarios, append '_{category_name}' to 'test' as the image_set.
        clip_ids, _pids = self._get_data_ids(image_set, params)
        
        # Commented out temporarily TODO
        # if len(set_ids) != 6: # Not equal when data_split_type is default.
        #     print("Considering the sets",set_ids,"for",image_set) # data_split_type is 'default'
        # else:
        #     category = image_set.split('_')[1]
        #     print("Considering pedestrians of category - {} for testing.".format(category))

        # if '_' in image_set:
        #     # True if image_set is changed for the certain categories.
        #     # Now that we've gotten the ped_ids we want, we can make image_set one of 'train', 'val', 'test'.
        #     image_set = image_set.split('_')[0]

        # set_ids, _pids = self._get_data_ids(image_set, params)

        for clip in clip_ids:
            img_width = annotations[clip]['width']
            pid_annots = annotations[clip]['pedestrian_annotations']
            vid_annots = annotations[clip]['vehicle_annotations']
            for pid in sorted(pid_annots):
                if params['data_split_type'] != 'default' and pid not in _pids:
                    continue
                num_pedestrians += 1
                frame_ids = pid_annots[pid]['frames']
                boxes = pid_annots[pid]['bbox']
                images = [self._get_image_path(clip, f) for f in frame_ids]

                if height_rng[0] > 0 or height_rng[1] < float('inf'):
                    images, boxes, frame_ids = self._height_check(height_rng,
                                                                        frame_ids, boxes,
                                                                            images)

                if len(boxes) / seq_stride < params['min_track_size']:
                    continue

                if sq_ratio:
                    boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                image_seq.append(images[::seq_stride])
                box_seq.append(boxes[::seq_stride])
                center_seq.append([self._get_center(b) for b in boxes][::seq_stride])

                ped_ids = [[pid]] * len(boxes)
                pids_seq.append(ped_ids[::seq_stride])

                assert len(image_seq) == len(box_seq)
                assert len(image_seq) == len(center_seq)

                obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                angv_seq.append([[vid_annots[i]['ang_vel']] for i in frame_ids][::seq_stride])

                # accT_seq.append([[vid_annots[i]['accT']] for i in frame_ids][::seq_stride])
                # obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                # accX_seq.append([[vid_annots[i]['accX']] for i in frame_ids][::seq_stride])
                # accY_seq.append([[vid_annots[i]['accY']] for i in frame_ids][::seq_stride])
                # accZ_seq.append([[vid_annots[i]['accZ']] for i in frame_ids][::seq_stride])
                    

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'obd_speed': obds_seq,
                'angular_velocity': angv_seq}
    
    # TODO - Add the vehicle annotations later
                # 'obd_speed': obds_seq,
                # 'accT': accT_seq,
                # 'accX': accX_seq,
                # 'accY': accY_seq,
                # 'accZ': accZ_seq}

    