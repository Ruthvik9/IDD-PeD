"""
Interface for the IDD-Pedestrian dataset:

Ruthvik.

"""
import cv2
import csv
import sys
import pdb
import pickle
import random
#import imageio
import numpy as np
# import seaborn as sns
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import makedirs, listdir,environ
from collections import Counter
from sklearn.model_selection import train_test_split, KFold
from os.path import join, abspath, isfile, isdir,basename, normpath,exists


# Set a global theme for Seaborn
# sns.set_theme(style="whitegrid")

class IDDPedestrian():
    def __init__(self, regen_database=False, data_path=''):
        """
        IDD-Pedestrian Class constructor
        :param regen_database: Whether generate the database or not
        :param data_path: The path to the root directory of the dataset.
        """
        self._year = '2024'
        self._name = 'iddp'
        self._image_ext = '.png'
        self._regen_database = regen_database

        # Paths
        self._iddp_path = data_path if data_path else self._get_default_path()
        assert isdir(self._iddp_path), \
            'The provided path {} does not exist. Kindly recheck the path provided'.format(self._iddp_path)

        self._annotation_path = join(self._iddp_path,'annotations')
        self._annotation_attributes_path = join(self._iddp_path, 'annotations_attributes')
        # NOOTE - The attributes are also in the annotations files and not in a separate folder so remove later.
        self._annotation_vehicle_path = join(self._iddp_path,'annotations_vehicle')

        self._videos_path = join(self._iddp_path,'videos')
        self._images_path = join(self._iddp_path,'images')

    # Path generators
    @property
    def cache_path(self):
        """
        Generates a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._iddp_path, 'data_cache'))
        if not isdir(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_default_path(self):
        """
        Returns the default path where the iddp dataset is expected to be installed.
        """
        return join('data','IDDPedestrian')

    def _get_image_set_ids(self, image_set):
        """
        Returns default image set ids
        :param image_set: Image set split
        :return: Set ids of the image set
        """
        # image_set_nums = {'train': ['set01', 'set02', 'set04'],
        #                   'val': ['set05', 'set06'],
        #                   'test': ['set03'],
        #                   'all': ['set01', 'set02', 'set03',
        #                           'set04', 'set05', 'set06']}
        image_set_nums = {'train': ['gp_set_0001','gp_set_0002','gp_set_0004','gp_set_0006','gp_set_0007'],
                          'val': ['gp_set_0001','gp_set_0002','gp_set_0004','gp_set_0006','gp_set_0007'],
                          'test': ['gp_set_0003','gp_set_0005','gp_set_0008','gp_set_0009'],
                          'all': ['gp_set_0001', 'gp_set_0002','gp_set_0003','gp_set_0004','gp_set_0005','gp_set_0006','gp_set_0007','gp_set_0008','gp_set_0009']}
        return image_set_nums[image_set]

    def _get_image_path(self, sid, vid, fid):
        """
        Generates and returns the image path given ids
        :param sid: Set id
        :param vid: Video id
        :param fid: Frame id
        :param camid: Camera id
        :return: Return the path to the given image
        """
        cam = sid.split('_')[0]
        if cam == "gp":
            cam = "gopro"
        elif cam == "d":
            cam = "ddpai"
        return join(self._images_path,cam, sid, vid,
                    '{:05d}.png'.format(fid))

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

    def process_xml_files(self,directory):
        """
        Processes the XML files in the given directory and generates a csv file of annotated frames
        :param directory: The directory containing the XML files
        """
        xml_files = [f for f in listdir(directory) if f.endswith('.xml')]
        set_name = basename(normpath(directory))
        print("Processing set", set_name)

        if exists(join(directory,'{}_annotated_frames.csv'.format(set_name))):
            raise Exception('{}_annotated_frames.csv already exists. Delete the file and try again.'.format(set_name))
        
        with open(join(directory,'{}_annotated_frames.csv'.format(set_name)), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for xml_file in xml_files:
                xml_path = join(directory, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                name = root.find('meta/task/name').text
                if '.MP4' in name:
                    name = name.replace('.MP4','')
                elif '.mp4' in name:
                    name = name.replace('.mp4','')

                frames = set()
                for track in root.findall('track'):
                    if track.get('label') != 'pedestrian':
                        continue
                    for box in track.findall('box'):
                        frame = int(box.get('frame'))
                        outside = box.get('outside')
                        if outside == '0':
                            frames.add(frame)

                frames = sorted(frames)
                row = [name, len(frames)] + frames
                writer.writerow(row)

    def create_annotated_frame_numbers(self,annotation_root_path):
        """
        Generates and saves a csv file of annotated frame numbers for each video in all the sets.
        This csv file will later be used for extracting images from the videos.
        :param annotation_path: The annotation root directory.
        """
        cams = [f for f in listdir(annotation_root_path)]
        print("Processing data from the camera", cams)
        for cam in cams:
            annotation_path_new = join(annotation_root_path, cam)
            sets = [f for f in listdir(annotation_path_new)]
            print("Processing sets", sets,"for camera",cam)
            for set_id in sets:
                set_path = join(annotation_path_new, set_id)
                self.process_xml_files(set_path)       


    # Image processing helpers
    def get_annotated_frame_numbers(self, set_id):
        """
        Returns a dictionary of videos and annotated frames for each video in the given set
        :param set_id: Set to retrieve annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<annotated_frame_id_0>,... <annotated_frame_id_n>]}
        """
        print("Generating annotated frame numbers for", set_id)
        cam = set_id.split('_')[0]
        if cam == "gp":
            cam = "gopro"
        elif cam == "d":
            cam = "ddpai"
        annotated_frames_file = join(self._iddp_path, "annotations",cam,set_id, set_id + '_annotated_frames.csv')
        # If the file exists, load from the file
        if isfile(annotated_frames_file):
            with open(annotated_frames_file, 'rt') as f:
                annotated_frames = {x.split(',')[0]:
                                        [int(fr) for fr in x.split(',')[1:]] for x in f.readlines()}
            return annotated_frames
        else:
            # Generate annotated frame ids for each video
            raise Exception('{} does not exist. Generate the file first if  it doesn\'t already exist using create_annotated_frame_numbers()'.format(annotated_frames_file))


    def get_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and  frames for each video in the give set
        :param set_id: Set to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<frame_id_0>,... <frame_id_n>]}
        """
        print("Generating frame numbers for", set_id)
        frame_ids = {v.split('_annt.xml')[0]: [] for v in sorted(listdir(join(self._annotation_path,
                                                                              set_id))) if
                     v.endswith("annt.xml")}
        for vid, frames in sorted(frame_ids.items()):
            path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
            tree = ET.parse(path_to_file)
            num_frames = int(tree.find("./meta/task/size").text)
            frames.extend([i for i in range(num_frames)])
            frames.insert(0, num_frames)
        return frame_ids

    # def create_pose_dict(self,bbox_thres=80,min_track=0,occlusion_tolerance=1,db_path="iddp_database.pkl"):
    #     """
    #     Create a dictionary containing info of which images and pedestrians
    #     are going to have their pose extracted.
    #     :param min_track: Min. track length of sequences.
    #     :param occlusion_tolerance: If 1, consider frames partially occluded or not 
    #                                 occluded. If 0, consider only frames which are not occlused (good for 3d  pose.)
    #     :param db_path: Path to the pickle file.
    #     """
    #     with open(db_path,"rb") as f:
    #         db = pickle.load(f)
        
    #     pose_db = {}
    #     for set_id in db.keys():
    #     # for set_id in ['gp_set_0003']:
    #         pose_db[set_id] = {}
    #         for vid_id in db[set_id]:
    #             pose_db[set_id][vid_id] = {}
    #             for pid in db[set_id][vid_id]['pedestrian_annotations'].keys():
    #                 frames = np.array(db[set_id][vid_id]['pedestrian_annotations'][pid]['frames'])
    #                 occlusion = np.array(db[set_id][vid_id]['pedestrian_annotations'][pid]['occlusion'])
    #                 bbox = np.array(db[set_id][vid_id]['pedestrian_annotations'][pid]['bbox'])
    #                 if pid == None:
    #                     print(f"Um, what {pid}")
    #                     break
    #                 # Account for pedestrian being a child, in which case bbox height is less to begin with.
    #                 # Account for the face that for some people, attributes are not matched yet.
    #                 if 'attributes' in db[set_id][vid_id]['pedestrian_annotations'][pid].keys(): 
    #                     age = db[set_id][vid_id]['pedestrian_annotations'][pid]['attributes']['age']
    #                     if age == 0: # Pedestrian is a child
    #                         threshold = int(bbox_thres * 0.6)
    #                     else:
    #                         threshold = bbox_thres
    #                 else:
    #                     threshold = bbox_thres

    #                 if occlusion_tolerance == 1:
    #                     indices = np.where(occlusion <= 1)[0].tolist()
    #                     frames = frames[indices]
    #                     bbox = bbox[indices]
    #                     assert len(frames) == len(bbox)
    #                     if len(frames) < min_track:
    #                         continue # Don't get this person's pose info.
    #                     for frame,box in zip(frames,bbox):
    #                         if abs(box[1]-box[3]) < threshold and abs(box[0] - box[2]) < threshold:
    #                             continue # Don't include the person in this frame since bbox is too small.
    #                         if frame not in pose_db[set_id][vid_id].keys():
    #                             pose_db[set_id][vid_id][frame] = [(pid,box.tolist())]
    #                         else:
    #                             pose_db[set_id][vid_id][frame].extend([(pid,box.tolist())])

    #                 elif occlusion_tolerance == 0:
    #                     indices = np.where(occlusion < 1)[0].tolist()
    #                     frames = frames[indices]
    #                     bbox = bbox[indices]
    #                     assert len(frames) == len(bbox)
    #                     if len(frames) < min_track:
    #                         continue # Don't get this person's pose info.
    #                     for frame,box in zip(frames,bbox):
    #                         if abs(box[1]-box[3]) < threshold and abs(box[0] - box[2]) < threshold:
    #                             continue # Don't include the person in this frame since bbox is too small.
    #                         if frame not in pose_db[set_id][vid_id].keys():
    #                             pose_db[set_id][vid_id][frame] = [(pid,box.tolist())]
    #                         else:
    #                             pose_db[set_id][vid_id][frame].extend([(pid,box.tolist())])

    #     with open("iddp_pose.pkl","wb") as f:
    #         pickle.dump(pose_db,f)

        # return pose_db

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

    
    # def save_pose_images(self,extract_frame_type='annotated',resize_shape=480,pose_pickle="iddp_pose.pkl",debug=False):
    #     with open(pose_pickle,"rb") as f:
    #                 pose_db = pickle.load(f)
    #     set_folders = [f for f in sorted(listdir(join(self._videos_path,"gopro")))]
    #     pose_bbox = {}
    #     # set_folders = ["gp_set_0003"]
    #     for set_id in set_folders:
    #         pose_bbox[set_id] = {}
    #         print('Extracting pose frames from', set_id)
    #         set_folder_path = join(self._videos_path,"gopro",set_id)
    #         if extract_frame_type == 'annotated':
    #             extract_frames = self.get_annotated_frame_numbers(set_id)
    #         else:
    #             raise NotImplementedError

    #         set_images_path = join(self._iddp_path, "poseimages", set_id)
    #         for vid, frames in sorted(extract_frames.items()):
    #             pose_bbox[set_id][vid] = {}
    #             pose_bbox[set_id][vid]['pedestrian_annotations'] = {}
    #             print(vid)
    #             video_images_path = join(set_images_path, vid)
    #             num_frames = frames[0]
    #             frames_list = frames[1:]
    #             # if exists(video_images_path):
    #             #     print(f"Folder {video_images_path} already exists. Skipping...")
    #             #     continue
    #             if not isdir(video_images_path):
    #                 makedirs(video_images_path)
    #             vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
    #             success, image = vidcap.read()
    #             frame_num = 0
    #             img_count = 0
            
    #             if not success:
    #                 print('Failed to open the video {}'.format(vid))
    #             while success:
    #                 if frame_num in frames_list:
    #                     if frame_num in pose_db[set_id][vid].keys():
    #                         # Logic to get images.
    #                         for e in pose_db[set_id][vid][frame_num]: # Create images for every pedestrian in frame_num
    #                             ped_id = e[0]
    #                             if ped_id == None:
    #                                 print("What on earth")
    #                                 continue
    #                             bbox = e[1]

                                
    #                             x1, y1, x2, y2 = map(int,bbox)
    #                             if (y2 - y1) <= 0 or (x2 - x1) <= 0:
    #                                 print(f"{x1,y1,x2,y2} faulty for pedestrian {ped_id} in {set_id},{vid}")
    #                                 continue
    #                             if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0] or x2 <= x1 or y2 <= y1:
    #                                 print(f"{x1,y1,x2,y2} faulty for pedestrian {ped_id} in {set_id},{vid},{frame_num}")
    #                                 continue
    #                             cropped_image = image[y1:y2, x1:x2]

    #                             # Calculate aspect ratio
    #                             h, w = cropped_image.shape[:2]
    #                             aspect_ratio = w / h
                                
    #                             if ped_id not in pose_bbox[set_id][vid]['pedestrian_annotations'].keys():
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id] = {}
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id]['frames'] = []
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id]['bbox'] = []
    #                                 bbox_new = self.calculate_transformed_bbox(cropped_image.shape[:2],resize_shape)
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id]['frames'].append(frame_num)
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id]['bbox'].append(bbox_new)
    #                             else:
    #                                 # No need to create the dictionary since it has already been created above.
    #                                 bbox_new = self.calculate_transformed_bbox(cropped_image.shape[:2],resize_shape)
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id]['frames'].append(frame_num)
    #                                 pose_bbox[set_id][vid]['pedestrian_annotations'][ped_id]['bbox'].append(bbox_new)

    #                             # Calculate new width and height
    #                             if h > w:
    #                                 new_h, new_w = resize_shape, int(resize_shape * aspect_ratio)
    #                             else:
    #                                 new_w, new_h = resize_shape, int(resize_shape / aspect_ratio)

    #                             # Resize image
    #                             resized_image = cv2.resize(cropped_image, (new_w, new_h))
                                

    #                             # Pad the image to make it square
    #                             delta_w = resize_shape - new_w
    #                             delta_h = resize_shape - new_h
    #                             top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    #                             left, right = delta_w // 2, delta_w - (delta_w // 2)
    #                             color = [0, 0, 0]
    #                             final_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    #                             # Save the image
    #                             filename = join(video_images_path, f"{frame_num:05d}_{ped_id}.png")
    #                             if debug == False:
    #                                 cv2.imwrite(filename, final_image)
    #                     self.update_progress(img_count / num_frames)
    #                     img_count += 1
    #                     # if not isfile(join(video_images_path, "%05.f.png") % frame_num):
    #                     #     cv2.imwrite(join(video_images_path, "%05.f.png") % frame_num, image)
    #                 success, image = vidcap.read()
    #                 frame_num += 1
    #             if num_frames != img_count:
    #                 print('num images don\'t match {}/{}'.format(num_frames, img_count))
    #             print('\n')
        
    #     with open("iddp_bbox_database.pkl","wb") as f:
    #         pickle.dump(pose_bbox,f)

    # def extract_detection_images(self,train_f,test_f):
    #     """
    #     Extract images containing objects relevant to object detection. If multiple objects are present in the image,
    #     obviously only one image will be extracted for all of them.
    #     """
    #     db = self.generate_database()

    #     # Extract training images
    #     train_img_dir = "mmdetection/data/IDDPedestrian/images/train/"
    #     # Create the directory if it doesn't exist
    #     makedirs(train_img_dir, exist_ok=True)
    #     train_sets=self._get_image_set_ids('train')
    #     for sid in train_sets:
    #         set_folder_path = join(self._videos_path,'gopro',sid)
    #         for vid in db[sid].keys():
    #             print("Extracting frames for object detection from video",vid)
    #             vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
    #             # Get the total number of frames in the video
    #             total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    #             # Create a progress bar with the total number of frames
    #             progress_bar = tqdm(total=total_frames, unit='frames')

    #             success, image = vidcap.read()
    #             frame_num = 0
    #             if not success:
    #                 raise Exception('Failed to open the video {}'.format(vid))
    #             while success:
    #                 img_name = sid + '_' + vid + '_' + f'{frame_num:05d}' + '.png'
    #                 if img_name in train_f:
    #                     if not isfile(join(train_img_dir, img_name)):
    #                         cv2.imwrite(join(train_img_dir, img_name), image)
    #                 success, image = vidcap.read()
    #                 frame_num += 1
    #                 # Update the progress bar
    #                 progress_bar.update(1)
    #             progress_bar.close()
    #             print('\n')

    #     # Extract test/val images
    #     test_img_dir = "mmdetection/data/IDDPedestrian/images/val/"
    #     # Create the directory if it doesn't exist
    #     makedirs(test_img_dir, exist_ok=True)
    #     test_sets=self._get_image_set_ids('test')
    #     for sid in test_sets:
    #         set_folder_path = join(self._videos_path,'gopro',sid)
    #         for vid in db[sid].keys():
    #             print("Extracting frames for object detection from video",vid)
    #             vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
    #             # Get the total number of frames in the video
    #             total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    #             # Create a progress bar with the total number of frames
    #             progress_bar = tqdm(total=total_frames, unit='frames')
    #             success, image = vidcap.read()
    #             frame_num = 0
    #             if not success:
    #                 raise Exception('Failed to open the video {}'.format(vid))
    #             while success:
    #                 img_name = sid + '_' + vid + '_' + f'{frame_num:05d}' + '.png'
    #                 if img_name in test_f:
    #                     if not isfile(join(test_img_dir, img_name)):
    #                         cv2.imwrite(join(test_img_dir, img_name), image)
    #                 success, image = vidcap.read()
    #                 frame_num += 1
    #                 # Update the progress bar
    #                 progress_bar.update(1)
    #             progress_bar.close()
    #             print('\n')

    def check_frames(self,video_images_path, frames_list):
        missing_frames = []
        for frame in frames_list:
            frame_filename = f"{int(frame):05d}.png"
            if not exists(join(video_images_path, frame_filename)):
                missing_frames.append(frame)
        return missing_frames



    def extract_and_save_images(self, extract_frame_type='annotated',lib='opencv'):
        """
        Extracts images from clips and saves on hard drive
        :param extract_frame_type: Whether to extract 'all' frames or only the ones that are 'annotated'
                             NOOTE - To do: mention to disc spaces consumed.
        :param lib: The library to use for extracting images. Options are 'imageio' and 'opencv'.
        You can use 'opencv' for faster extraction if you have ffmpeg installed. Else, use the slower lib 'imageio'.
        """
        # cams = [f for f in listdir(self._videos_path)]
        cams = ['gopro']
        for cam in cams:
            print("############################################")
            print("Extracting and saving images for camera", cam)
            print("############################################")
            set_folders = ['gp_set_0001','gp_set_0002','gp_set_0003'] # To only get the sets which have annotations.
            for set_id in set_folders:
                print('Extracting frames from', set_id)
                set_folder_path = join(self._videos_path,cam,set_id)
                if extract_frame_type == 'annotated':
                    extract_frames = self.get_annotated_frame_numbers(set_id)
                else:
                    extract_frames = self.get_frame_numbers(set_id) # NOOTE - to change. the get_frame_numbers func ie.

                set_images_path = join(self._iddp_path,"images",cam,set_id)
                for vid, frames in sorted(extract_frames.items()):
                    print(vid)
                    num_frames = frames[0]
                    frames_list = frames[1:]
                    video_images_path = join(set_images_path, vid)
                    if exists(video_images_path):
                        missing_frames = self.check_frames(video_images_path, frames_list)
                        if not missing_frames:
                            print(f"Images for {set_id} and {vid} already exist and all frames are present. Skipping...")
                            continue
                        else:
                            print(f"Folder for {set_id} and {vid} exists, but not all frames match.")
                            print(f"Missing frames: {missing_frames}")
                            print("Proceeding with extraction...")
                    if not isdir(video_images_path):
                        makedirs(video_images_path)
                    
                    if lib == 'imageio':
                        reader = imageio.get_reader(join(set_folder_path,vid+'.MP4'))
                        #vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
                        #success, image = vidcap.read()
                        frame_num = 0
                        img_count = 0
                        #if not success:
                        #    raise Exception('Failed to open the video {}'.format(vid))
                        for frame in reader:
                            if frame_num in frames_list:
                                self.update_progress(img_count / num_frames)
                                img_count += 1
                                if not isfile(join(video_images_path, "%05d.png") % frame_num):
                                    imageio.imwrite(join(video_images_path, "%05d.png") % frame_num, frame)
                            #success, image = vidcap.read()
                            frame_num += 1
                        if num_frames != img_count:
                            print(f'num images {num_frames} don\'t match image count {img_count}')
                        print('\n')
                    elif lib == 'opencv':
                        vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
                        success, image = vidcap.read()
                        frame_num = 0
                        img_count = 0
                        if not success:
                            raise Exception('Failed to open the video {}'.format(vid))
                        while success:
                            if frame_num in frames_list:
                                self.update_progress(img_count / num_frames)
                                img_count += 1
                                if not isfile(join(video_images_path, "%05d.png") % frame_num):
                                    cv2.imwrite(join(video_images_path, "%05d.png") % frame_num, image)
                            success, image = vidcap.read()
                            frame_num += 1
                        if num_frames != img_count:
                            print(f'num images {num_frames} don\'t match image count {img_count}')
                        print('\n')

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
                   'state': {'red': 0, 'orange': 1, 'green': 2},
                   'gender': {'male': 0, 'female': 1, 'default': 2},
                   'category': {'interaction': 0, 'jaywalking': 1, 'crowded': 2, 'crossing': 3, 'ATS': 4, 'stationary': 5, 'default': 6}}

        
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
                   'state': {0: 'red', 1: 'orange', 2: 'green'},
                   'gender': {0: 'male', 1: 'female', 2: 'default'},
                   'category': {0: 'interaction', 1: 'jaywalking', 2: 'crowded', 3: 'crossing', 4: 'ATS', 5: 'stationary', 6: 'default'}}

        return map_dic[label_type][value]

    def _get_annotations(self, setid, vid):
        """
        Generates a dictionary of annotations by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of annotations
        """
        cam = setid.split('_')[0]
        if cam == "gp":
            cam = "gopro"
        elif cam == "d":
            cam = "ddpai"
        path_to_file = join(self._annotation_path,cam, setid, vid +'.xml')

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

    def _get_vehicle_attributes(self, setid, vid):
        """
        Generates a dictionary of vehicle attributes by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of vehicle attributes (obd sensor recording)
        """
        cam = setid.split('_')[0]
        if cam == "gp":
            cam = "gopro"
        elif cam == "d":
            cam = "ddpai"
        path_to_file = join(self._annotation_vehicle_path,cam, setid, vid + '_obd.xml')
        tree = ET.parse(path_to_file)

        veh_attributes = {}
        frames = tree.findall("./frame")

        for f in frames:
            dict_vals = {k: float(v) for k, v in f.attrib.items() if k != 'id'}
            veh_attributes[int(f.get('id'))] = dict_vals

        return veh_attributes

    def generate_database(self):
        """
        Generates and saves a "database" of the IDDP dataset by integrating all annotations. Basically it's just a *dictionary* 
        containing all annotations in the dataset.

        Dictionary structure:
        'set_id'(str): {
            'vid_id'(str): {
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
        print("Generating database for the IDDPedestrian dataset...")

        cache_file = join(self.cache_path, 'iddp_database.pkl')
        if isfile(cache_file) and not self._regen_database:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('IDDP annotations loaded from {}'.format(cache_file))
            return database

        print("####################################################################################")
        print("Database cache file not found. Generating database for the IDD-Pedestrian dataset...")
        print("####################################################################################")
        
        # Path to the folder annotations
        cams = [f for f in listdir(self._annotation_path)]
        set_ids = []
        for cam in cams:
            set_ids.extend([f for f in listdir(join(self._annotation_path,cam))])

        print("The following sets are found in the dataset:", set_ids)

        # Read the content of set folders
        database = {}
        for setid in set_ids:
            if setid.split('_')[0] == "gp":
                cam = "gopro"
            elif setid.split('_')[0] == "d":
                cam = "ddpai"
            video_ids = [v.split(".xml")[0] for v in sorted(listdir(join(self._annotation_path,cam,setid))) if v.endswith(".xml")]
            database[setid] = {}
            for vid in video_ids:
                print('Getting annotations for %s, %s' % (setid, vid))
                database[setid][vid] = self._get_annotations(setid, vid)
                vid_attributes = self._get_ped_attributes(setid, vid)
                database[setid][vid]['vehicle_annotations'] = self._get_vehicle_attributes(setid, vid)
                for ped in database[setid][vid]['pedestrian_annotations']:
                    try:
                        database[setid][vid]['pedestrian_annotations'][ped]['attributes'] = vid_attributes[ped]
                    except KeyError:
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        print("Key {} not found, kindly check".format(ped))
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        continue

        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database

    # def get_data_stats(self,plot=False):
    #     """
    #     Generates statistics for the dataset
    #     :param: plot - Whether to plot the statistics or not. Default - False.
    #     """
    #     annotations = self.generate_database()

    #     set_count = len(annotations.keys())

    #     ped_count = 0
    #     ped_box_count = 0
    #     video_count = 0
    #     total_frames = 0
    #     age = {'child': 0, 'teenager': 0, 'adult': 0, 'senior': 0}
    #     crossing = {'no': 0, 'yes': 0}
    #     intersection_type = {'NI': 0, 'U-turn': 0, 'T-right': 0, 'T-left': 0, 'four-way': 0, 'Y-intersection': 0}
    #     motion_direction = {'OW': 0, 'TW': 0}
    #     signalized_type = {'N/A': 0, 'C': 0, 'S': 0, 'CS': 0}
    #     road_type = {'main': 0, 'secondary': 0, 'street': 0, 'lane': 0}
    #     location_type = {'urban': 0, 'rural': 0, 'commercial': 0, 'residential': 0}
        
    #     crossing_motive = {'yes': 0, 'maybe': 0, 'no': 0}
    #     carrying_object = {'none': 0, 'small': 0, 'large': 0}
    #     crosswalk_usage = {'yes': 0, 'no': 0, 'partial': 0, 'N/A': 0}


    #     num_frames_prior_event = {}
    #     group_size_count = {}

    #     occlusion = {'None': 0, 'Part': 0, 'Full': 0}
    #     crossing_behavior = {'CU': 0, 'CFU': 0, 'CD': 0, 'CFD': 0, 'N/A': 0, 'CI': 0}
    #     traffic_interaction = {'WTT': 0, 'HG': 0, 'Other': 0, 'N/A': 0}
    #     pedestrian_activity = {'Walking': 0, 'MS': 0, 'N/A': 0}
    #     attention_indicators = {'LOS': 0, 'FTT': 0, 'NL': 0, 'DB': 0}
    #     social_dynamics = {'GS': 0, 'CFA': 0, 'AWC': 0, 'N/A': 0}
    #     stationary_behavior = {'Sitting': 0, 'Standing': 0, 'IWA': 0, 'Other': 0, 'N/A': 0}

    #     state = {'red': 0, 'orange': 0, 'green': 0} # Only for traffic lights. We'll count the num frames for each.


    #     traffic_obj_types = {'vehicle': {'car': 0, 'motorcycle': 0, 'bicycle': 0, 'auto': 0, 'bus': 0, 'cart': 0, 'truck': 0, 'other': 0},
    #                          'traffic_light': {'pedestrian': 0, 'vehicle': 0},
    #                          'crosswalk': 0,
    #                          'bus_station': 0}
        
        
    #     traffic_box_count = {'vehicle': 0, 'traffic_light': 0, 'crosswalk': 0, 'bus_station': 0}

    #     for sid, vids in annotations.items():
    #         video_count += len(vids)
    #         for vid, annots in vids.items():
    #             total_frames += annots['num_frames']
    #             for trf_ids, trf_annots in annots['traffic_annotations'].items():
    #                 obj_class = trf_annots['obj_class']
    #                 traffic_box_count[obj_class] += len(trf_annots['frames'])
    #                 if obj_class in ['traffic_light', 'vehicle']:
    #                     obj_type = trf_annots['obj_type']
    #                     traffic_obj_types[obj_class][self._map_scalar_to_text(obj_class, obj_type)] += 1
    #                     # Get stats for traffic light states
    #                     if obj_class == "traffic_light":
    #                         trf_states = trf_annots['state']
    #                         state_counter = Counter(trf_states)
    #                         for key,value in state_counter.items():
    #                             state[self._map_scalar_to_text('state',key)] += value
    #                 else:
    #                     traffic_obj_types[obj_class] += 1 # For crosswalk, bus_station.

    #             for ped_ids, ped_annots in annots['pedestrian_annotations'].items():
    #                 ped_count += 1
    #                 ped_box_count += len(ped_annots['frames'])

    #                 occl = ped_annots['occlusion']
    #                 occl_counter = Counter(occl)
    #                 for key,value in occl_counter.items():
    #                     occlusion[self._map_scalar_to_text('occlusion',key)] += value

    #                 cb = ped_annots['behavior']['CrossingBehavior']
    #                 cb_counter = Counter(cb)
    #                 for key,value in cb_counter.items():
    #                     crossing_behavior[self._map_scalar_to_text('CrossingBehavior',key)] += value

                    
    #                 ti = ped_annots['behavior']['TrafficInteraction']
    #                 ti_counter = Counter(ti)
    #                 for key,value in ti_counter.items():
    #                     traffic_interaction[self._map_scalar_to_text('TrafficInteraction',key)] += value

    #                 pa = ped_annots['behavior']['PedestrianActivity']
    #                 pa_counter = Counter(pa)
    #                 for key,value in pa_counter.items():
    #                     pedestrian_activity[self._map_scalar_to_text('PedestrianActivity',key)] += value

    #                 ai = ped_annots['behavior']['AttentionIndicators']
    #                 ai_counter = Counter(ai)
    #                 for key,value in ai_counter.items():
    #                     attention_indicators[self._map_scalar_to_text('AttentionIndicators',key)] += value

    #                 sd = ped_annots['behavior']['SocialDynamics']
    #                 sd_counter = Counter(sd)
    #                 for key,value in sd_counter.items():
    #                     social_dynamics[self._map_scalar_to_text('SocialDynamics',key)] += value

    #                 sb = ped_annots['behavior']['StationaryBehavior']
    #                 sb_counter = Counter(sb)
    #                 for key,value in sb_counter.items():
    #                     stationary_behavior[self._map_scalar_to_text('StationaryBehavior',key)] += value

    #                 first_frame = ped_annots['frames'][0]
    #                 last_frame = ped_annots['frames'][-1]
    #                 try:
    #                     temp = ped_annots['attributes'] # temporary check since currently some pedestrians are wrongly id'ed.
    #                 except:
    #                     continue
    #                 crossing_point = ped_annots['attributes']['crossing_point']
    #                 if crossing_point > last_frame or crossing_point < first_frame:
    #                     print(f"Crossing point {crossing_point} for pedestrian {ped_ids} in video {vid} is out of range {first_frame} - {last_frame}")
    #                     continue
    #                 if ped_annots['attributes']['crossing'] == 1: # Pedestrian is crossing
    #                     prior_event = crossing_point - first_frame
    #                     if prior_event not in num_frames_prior_event.keys():
    #                         num_frames_prior_event[prior_event] = 1
    #                     else:
    #                         num_frames_prior_event[prior_event] += 1
    #                # So, if num_frames_prior_event[x] = y, it means there are y pedestrians tracks
    #                 # with x frames before event.
    #                 # Sort the dictionary by key.
    #                 num_frames_prior_event = dict(sorted(num_frames_prior_event.items(), key=lambda x: x[0]))

    #                 # Get group size counts.
    #                 group_size = ped_annots['attributes']['group_size']
    #                 if group_size not in group_size_count.keys():
    #                     group_size_count[group_size] = 1
    #                 else:
    #                     group_size_count[group_size] += 1
    #                 group_size_count = dict(sorted(group_size_count.items(), key=lambda x:x[0]))


    #                 age[self._map_scalar_to_text('age', ped_annots['attributes']['age'])] += 1
    #                 if self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing']) == 'yes':
    #                     crossing['yes'] += 1
    #                 else:
    #                     crossing['no'] += 1                
    #                 intersection_type[
    #                     self._map_scalar_to_text('intersection_type', ped_annots['attributes']['intersection_type'])] += 1
    #                 motion_direction[self._map_scalar_to_text('motion_direction',
    #                                                            ped_annots['attributes']['motion_direction'])] += 1
    #                 signalized_type[self._map_scalar_to_text('signalized_type', ped_annots['attributes']['signalized_type'])] += 1
    #                 road_type[self._map_scalar_to_text('road_type', ped_annots['attributes']['road_type'])] += 1
    #                 location_type[self._map_scalar_to_text('location_type', ped_annots['attributes']['location_type'])] += 1
    #                 carrying_object[self._map_scalar_to_text('carrying_object', ped_annots['attributes']['carrying_object'])] += 1
    #                 crosswalk_usage[self._map_scalar_to_text('crosswalk_usage', ped_annots['attributes']['crosswalk_usage'])] += 1
    #                 crossing_motive[self._map_scalar_to_text('crossing_motive', ped_annots['attributes']['crossing_motive'])] += 1
        


    #     print('---------------------------------------------------------')
    #     print('Data stats of the IDDP Dataset')
    #     print('---------------------------------------------------------')
    #     print("Number of sets: %d" % set_count)
    #     print("Number of videos: %d" % video_count)
    #     print("Number of annotated frames: %d" % total_frames)
    #     print("Number of pedestrians %d" % ped_count)
    #     print("Age:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(age.items())))
    #     print("Signal type:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(signalized_type.items())))
    #     print("Motion direction:\n",
    #           '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(motion_direction.items())))
    #     print("crossing:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(crossing.items())))
    #     print("Intersection type:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(intersection_type.items())))
    #     print("Road type:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(road_type.items())))
    #     print("Location type:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(location_type.items())))
    #     print("Carrying object:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(carrying_object.items())))
    #     print("Crosswalk usage:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(crosswalk_usage.items())))
    #     print("Crossing motive:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(crossing_motive.items())))
    #     print("Group size distribution:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(group_size_count.items())))
    #     print("\nNumber of bounding boxes for different types of objects:\n")
    #     print("Number of pedestrian bounding boxes: %d" % ped_box_count)
    #     print("Number of traffic objects")
    #     for trf_obj, values in sorted(traffic_obj_types.items()):
    #         if isinstance(values, dict):
    #             print(trf_obj + ':\n', '\n '.join('{}: {}'.format(k, v) for k, v in sorted(values.items())),
    #                   '\n total: ', sum(values.values()))
    #         else:
    #             print(trf_obj + ': %d' % values)
    #     print("Number of traffic object bounding boxes:\n",
    #           '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_box_count.items())),
    #           '\n total: ', sum(traffic_box_count.values()))
        

    #     if plot == True:
    #         counts = [ped_count, ped_box_count, video_count, total_frames]
    #         labels = ['Pedestrian Count', 'Ped Box Count', 'Video Count', 'Total Frames']
    #         self.plot_basic_counts(counts, labels, 'basic_counts','plots')

    #         traffic_obj_types_plot = {}
    #         for key,value in traffic_obj_types.items():
    #             if key == 'vehicle':
    #                 for k,v in value.items():
    #                     traffic_obj_types_plot[f'{k}_v'] = v # For vehicle types.
    #             elif key == 'traffic_light':
    #                 for k,v in value.items():
    #                     traffic_obj_types_plot[f'{k}_t'] = v # For traffic light types.
    #             else:
    #                 traffic_obj_types_plot[key] = value # For crosswalk, bus_station.
            
    #         #all_dicts_str = ['age','crossing','intersection_type','motion_direction','signalized_type','road_type','location_type',\
    #         #        'crossing_motive','carrying_object','crosswalk_usage','occlusion','crossing_behavior','traffic_interaction','pedestrian_activity',\
    #         #        'attention_indicators','social_dynamics','stationary_behavior','state','traffic_box_count','traffic_obj_types_plot']

            

    #         all_dicts = {'age':age,'crossing':crossing,'intersection_type':intersection_type,'motion_direction':motion_direction,\
    #                 'signalized_type':signalized_type,'road_type':road_type,'location_type':location_type,'crossing_motive':crossing_motive,\
    #                 'carrying_object':carrying_object,'crosswalk_usage':crosswalk_usage,'occlusion':occlusion,'crossing_behavior':crossing_behavior,\
    #                 'traffic_interaction':traffic_interaction,'pedestrian_activity':pedestrian_activity,'attention_indicators':attention_indicators,\
    #                 'social_dynamics':social_dynamics,'stationary_behavior':stationary_behavior,'state':state,'traffic_box_count':traffic_box_count,\
    #                 'traffic_obj_types_plot':traffic_obj_types_plot}

    #         for title,var in all_dicts.items():
    #             self.plot_dict_with_colors(var, f'{title} Distribution', f'{title}', f'{title}_distribution','plots')
            
    #         time_series_dicts = {'num_frames_prior_event':num_frames_prior_event,'group_size_count':group_size_count}
    #         for title,var in time_series_dicts.items():
    #             self.plot_time_series(var, f'{title} Distribution', f'{title}', f'{title}_distribution','plots')

            



    # def plot_basic_counts(self,counts, labels, file_name,root_dir):
    #     plt.figure(figsize=(10, 6))
    #     ax = sns.barplot(x=labels, y=counts, palette="bright")
    #     ax.set_title('Basic Counts', fontsize=16)
    #     ax.set_ylabel('Count', fontsize=14)
    #     ax.set_xlabel('Categories', fontsize=14)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     ax.grid(True, linestyle='--', linewidth=0.5)
    #     for p in ax.patches:
    #         ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
    #                 ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
    #                     textcoords='offset points')
    #     plt.savefig(join(root_dir,f'{file_name}.png'), dpi=300, bbox_inches='tight')

    # def plot_dict_with_colors(self,data, title, xlabel, file_name,root_dir):
    #     keys = list(data.keys())
    #     values = list(data.values())
    #     colors = sns.color_palette("hsv", len(keys))  # You can choose any palette

    #     plt.figure(figsize=(10, 6))
    #     ax = plt.subplot()

    #     # Create the barplot
    #     bars = plt.bar(keys, values, color=colors)

    #     # Set titles and labels
    #     ax.set_title(title, fontsize=16)
    #     ax.set_ylabel('Count', fontsize=14)
    #     ax.set_xlabel(xlabel, fontsize=14)
        
    #     # Customize ticks
    #     if title == 'traffic_obj_types_plot Distribution':
    #         plt.xticks(fontsize=10,rotation=90) # Rotate the x labels for better visibility because there are too many.
    #     else:
    #         plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)

    #     # Add grid
    #     ax.grid(True, linestyle='--', linewidth=0.5)

    #     # Add annotations
    #     for bar in bars:
    #         yval = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
    #                 ha='center', va='bottom', fontsize=12, color='black')
    #     # Save the figure
    #     plt.savefig(join(root_dir,f'{file_name}.png'), dpi=300, bbox_inches='tight')

    # def plot_nested_dict(self,data, title,file_name,root_dir):
    #     categories = list(data.keys())
    #     sub_categories = list(data[categories[0]].keys())
    #     bar_width = 0.15
    #     r = np.arange(len(sub_categories))
        
    #     plt.figure(figsize=(12, 8))
    #     for i, category in enumerate(categories):
    #         values = [data[category][sub] for sub in sub_categories]
    #         plt.bar(r + bar_width * i, values, width=bar_width, edgecolor='white', label=category)

    #     plt.title(title, fontsize=16)
    #     plt.xticks([r + bar_width for r in range(len(sub_categories))], sub_categories, fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.xlabel('Categories', fontsize=14)
    #     plt.ylabel('Count', fontsize=14)
    #     plt.legend()
    #     plt.grid(True, linestyle='--', linewidth=0.5)  # Adding grid
    #     plt.savefig(join(root_dir,f'{file_name}.png'), dpi=300, bbox_inches='tight')

    
    # def plot_time_series(self,data, title, xlabel,file_name,root_dir):
    #     plt.figure(figsize=(12, 6))
    #     ax = sns.lineplot(x=list(data.keys()), y=list(data.values()), marker='o', sort=False, lw=2)
    #     ax.set_title(title, fontsize=16)
    #     ax.set_xlabel(xlabel, fontsize=14)
    #     ax.set_ylabel('Count', fontsize=14)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     ax.grid(True, linestyle='--', linewidth=0.5)  # Adding grid
    #     plt.savefig(join(root_dir,f'{file_name}.png'), dpi=300, bbox_inches='tight')

    
    


    def balance_samples_count(self, seq_data, label_type, random_seed=42):
        """
        Balances the number of positive and negative samples by randomly sampling
        from the more represented samples. Only works for binary classes.
        :param seq_data: The sequence data to be balanced.
        :param label_type: The lable type based on which the balancing takes place.
        The label values must be binary, i.e. only 0, 1.
        :param random_seed: The seed for random number generator.
        :return: Balanced data sequence.
        """
        for lbl in seq_data[label_type]:
            for i in lbl:
                if i[0] not in [0, 1]:
                    raise Exception("The label values used for balancing must be"
                                    " either 0 or 1")

        # balances the number of positive and negative samples
        print('---------------------------------------------------------')
        print("Balancing the number of positive and negative intention samples")

        gt_labels = [gt[0] for gt in seq_data[label_type]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        new_seq_data = {}
        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
            return seq_data
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(random_seed)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]
            # update the data
            for k in seq_data:
                seq_data_k = seq_data[k]
                if not isinstance(seq_data[k], list):
                    new_seq_data[k] = seq_data[k]
                else:
                    new_seq_data[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in new_seq_data[label_type]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(new_seq_data[label_type]) - num_pos_samples))
        return new_seq_data

    # Process pedestrian ids
    def _get_pedestrian_ids(self):
        """
        Returns all pedestrian ids
        :return: A list of pedestrian ids
        """
        annotations = self.generate_database()
        pids = []
        for sid in sorted(annotations):
            for vid in sorted(annotations[sid]):
                pids.extend(annotations[sid][vid]['ped_annotations'].keys())
        return pids

    def _get_random_pedestrian_ids(self, image_set, ratios=None, val_data=True, regen_data=False):
        """
        Generates and saves a random pedestrian ids
        :param image_set: The data split to return
        :param ratios: The ratios to split the data. There should be 2 ratios (or 3 if val_data is true)
        and they should sum to 1. e.g. [0.4, 0.6], [0.3, 0.5, 0.2]
        :param val_data: Whether to generate validation data
        :param regen_data: Whether to overwrite the existing data, i.e. regenerate splits
        :return: The random sample split
        """

        assert image_set in ['train', 'test', 'val']
        # Generates a list of behavioral xml file names for  videos
        cache_file = join(self.cache_path, "random_samples.pkl")
        if isfile(cache_file) and not regen_data:
            print("Random sample currently exists.\n Loading from %s" % cache_file)
            with open(cache_file, 'rb') as fid:
                try:
                    rand_samples = pickle.load(fid)
                except:
                    rand_samples = pickle.load(fid, encoding='bytes')
                assert image_set in rand_samples, "%s does not exist in random samples\n" \
                                                  "Please try again by setting regen_data = True" % image_set
                if val_data:
                    assert len(rand_samples['ratios']) == 3, "The existing random samples " \
                                                             "does not have validation data.\n" \
                                                             "Please try again by setting regen_data = True"
                if ratios is not None:
                    assert ratios == rand_samples['ratios'], "Specified ratios {} does not match the ones in existing file {}.\n\
                                                              Perform one of the following options:\
                                                              1- Set ratios to None\
                                                              2- Set ratios to the same values \
                                                              3- Regenerate data".format(ratios, rand_samples['ratios'])

                print('The ratios are {}'.format(rand_samples['ratios']))
                print("Number of %s tracks %d" % (image_set, len(rand_samples[image_set])))
                return rand_samples[image_set]

        if ratios is None:
            if val_data:
                ratios = [0.5, 0.4, 0.1]
            else:
                ratios = [0.5, 0.5]

        assert sum(ratios) > 0.999999, "Ratios {} do not sum to 1".format(ratios)
        if val_data:
            assert len(ratios) == 3, "To generate validation data three ratios should be selected"
        else:
            assert len(ratios) == 2, "With no validation only two ratios should be selected"

        print("################ Generating Random training/testing data ################")
        ped_ids = self._get_pedestrian_ids()
        print("Toral number of tracks %d" % len(ped_ids))
        print('The ratios are {}'.format(ratios))
        sample_split = {'ratios': ratios}
        train_samples, test_samples = train_test_split(ped_ids, train_size=ratios[0])
        print("Number of train tracks %d" % len(train_samples))

        if val_data:
            test_samples, val_samples = train_test_split(test_samples, train_size=ratios[1] / sum(ratios[1:]))
            print("Number of val tracks %d" % len(val_samples))
            sample_split['val'] = val_samples

        print("Number of test tracks %d" % len(test_samples))
        sample_split['train'] = train_samples
        sample_split['test'] = test_samples

        cache_file = join(self.cache_path, "random_samples.pkl")
        with open(cache_file, 'wb') as fid:
            pickle.dump(sample_split, fid, pickle.HIGHEST_PROTOCOL)
            print('pie {} samples written to {}'.format('random', cache_file))
        return sample_split[image_set]

    def _get_kfold_pedestrian_ids(self, image_set, num_folds=5, fold=1):
        """
        Generates kfold pedestrian ids
        :param image_set: Image set split
        :param num_folds: Number of folds
        :param fold: The given fold
        :return: List of pedestrian ids for the given fold
        """
        assert image_set in ['train', 'test'], "Image set should be either \"train\" or \"test\""
        assert fold <= num_folds, "Fold number should be smaller than number of folds"
        print("################ Generating %d fold data ################" % num_folds)
        cache_file = join(self.cache_path, "%d_fold_samples.pkl" % num_folds)

        if isfile(cache_file):
            print("Loading %d-fold data from %s" % (num_folds, cache_file))
            with open(cache_file, 'rb') as fid:
                try:
                    fold_idx = pickle.load(fid)
                except:
                    fold_idx = pickle.load(fid, encoding='bytes')
        else:
            ped_ids = self._get_pedestrian_ids()
            kf = KFold(n_splits=num_folds, shuffle=True)
            fold_idx = {'pid': ped_ids}
            count = 1
            for train_index, test_index in kf.split(ped_ids):
                fold_idx[count] = {'train': train_index.tolist(), 'test': test_index.tolist()}
                count += 1
            with open(cache_file, 'wb') as fid:
                pickle.dump(fold_idx, fid, pickle.HIGHEST_PROTOCOL)
                print('pie {}-fold samples written to {}'.format(num_folds, cache_file))
        print("Number of %s tracks %d" % (image_set, len(fold_idx[fold][image_set])))
        kfold_ids = [fold_idx['pid'][i] for i in range(len(fold_idx['pid'])) if i in fold_idx[fold][image_set]]
        return kfold_ids

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
            print("Going through set",sid)
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

    def _height_check(self, height_rng, frame_ids, boxes, images, occlusion):
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
        imgs, box, frames, occ = [], [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)
                imgs.append(images[i])
                frames.append(frame_ids[i])
                occ.append(occlusion[i])
        return imgs, box, frames, occ
    
    def _beh_filter(self,height_rng,frame_ids,boxes,images,occlusions,beh_labels,occluded=False):
        """
        Checks whether the bounding boxes are within a given height limit. If not, the data where
        the boxes aren't in the height range are ignored while creating the sequence.
        :param occluded: If fully occluded frames have to be included. Default = False.
        """
        if occluded == False:
            occlusion_val = 1
        else:
            occlusion_val = 2
        imgs, box, frames, occ, beh = [], [], [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])
            if height_rng[0] <= bbox_height <= height_rng[1] and occlusions[i] <= occlusion_val:
                box.append(b)
                imgs.append(images[i])
                frames.append(frame_ids[i])
                occ.append(occlusions[i])
                beh.append(beh_labels[i])
        return imgs, box, frames, occ, beh

    

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
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq, resolution_seq = [], []
        obds_seq, accT_seq, accX_seq, accY_seq, accZ_seq = [], [], [], [], []

        # For testing trajectory prediction for certain scenarios, append '_{category_name}' to 'test' as the image_set.
        set_ids, _pids = self._get_data_ids(image_set, params)
        
        if len(set_ids) != 9: # Only equal when data_split_type is not default.
            print("Considering the sets",set_ids,"for",image_set) # data_split_type is 'default'
        else:
            category = image_set.split('_')[1]
            print("Considering pedestrians of category - {} for testing.".format(category))

        if '_' in image_set:
            # True if image_set is changed for the certain categories.
            # Now that we've gotten the ped_ids we want, we can make image_set one of 'train', 'val', 'test'.
            image_set = image_set.split('_')[0]

        # set_ids, _pids = self._get_data_ids(image_set, params)

        vid_list = []
        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                vid_list.append(vid)


        all_peds = []
        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                pid_annots = annotations[sid][vid]['pedestrian_annotations']
                for pid in sorted(pid_annots):
                    all_peds.append(pid) # Use all these peds unless signalized type is specified


        try:
            if params['sig_type'] == 'y' and image_set == 'test':
                all_peds = ['gp_3277', 'gp_2485', 'gp_2486', 'gp_2490', 'gp_2520', 'gp_2523', 'gp_2524', 'gp_2526', 'gp_2527', 'gp_2528', 'gp_2802', 'gp_2804', 'gp_3050', 'gp_3051', 'gp_3061', 'gp_3062', 'gp_3064', 'gp_3073', 'gp_3171', 'gp_2638', 'gp_2644', 'gp_2645', 'gp_2707', 'gp_2819', 'gp_2948', 'gp_2949', 'gp_2957', 'gp_2962', 'gp_2988', 'gp_6944', 'gp_6945', 'gp_6947', 'gp_6969', 'gp_6971', 'gp_1649', 'gp_1650', 'gp_1682', 'gp_1683', 'gp_1711', 'gp_1713', 'gp_4049', 'gp_4052', 'gp_4067', 'gp_4068', 'gp_4069', 'gp_4070', 'gp_5748', 'gp_5750', 'gp_5751', 'gp_5758', 'gp_5763', 'gp_5797', 'gp_5858', 'gp_5860', 'gp_5863', 'gp_5864', 'gp_5868', 'gp_5870', 'gp_5974', 'gp_5975', 'gp_5977', 'gp_5980', 'gp_5981', 'gp_6514', 'gp_6515', 'gp_6530', 'gp_6532', 'gp_6069', 'gp_6072', 'gp_6074', 'gp_6076', 'gp_6078', 'gp_6079', 'gp_1728', 'gp_1731', 'gp_1732', 'gp_1733', 'gp_1735', 'gp_1736', 'gp_1737', 'gp_1740', 'gp_2683', 'gp_2929', 'gp_2933', 'gp_2938', 'gp_2939', 'gp_2941', 'gp_2946', 'gp_3503', 'gp_3506', 'gp_3509', 'gp_3511', 'gp_3513', 'gp_3515', 'gp_3520', 'gp_1668', 'gp_1670', 'gp_1672', 'gp_1674', 'gp_1678', 'gp_4208', 'gp_4227', 'gp_4230', 'gp_4030', 'gp_4031', 'gp_4035', 'gp_4036', 'gp_4039', 'gp_4040', 'gp_4043', 'gp_4047', 'gp_3279', 'gp_3284', 'gp_3009', 'gp_3013', 'gp_3088', 'gp_3092', 'gp_3112', 'gp_3212', 'gp_2964', 'gp_2967', 'gp_4007', 'gp_6877', 'gp_6878', 'gp_6879', 'gp_6880', 'gp_6906', 'gp_6908', 'gp_6909', 'gp_6910', 'gp_6911', 'gp_6912', 'gp_6917', 'gp_6918', 'gp_6919', 'gp_6920', 'gp_6922', 'gp_6936', 'gp_6937']
            elif params['sig_type'] == 'n' and image_set =='test':
                all_peds = ['gp_3252', 'gp_3256', 'gp_3260', 'gp_3264', 'gp_3266', 'gp_3267', 'gp_3273', 'gp_3287', 'gp_3289', 'gp_3294', 'gp_3296', 'gp_3299', 'gp_3302', 'gp_3304', 'gp_3307', 'gp_3313', 'gp_3320', 'gp_3325', 'gp_3326', 'gp_3334', 'gp_3338', 'gp_3339', 'gp_3341', 'gp_3342', 'gp_3343', 'gp_3344', 'gp_3347', 'gp_3348', 'gp_3352', 'gp_3356', 'gp_3362', 'gp_3365', 'gp_3367', 'gp_3368', 'gp_3369', 'gp_3372', 'gp_3373', 'gp_3374', 'gp_3381', 'gp_3382', 'gp_3384', 'gp_3386', 'gp_3389', 'gp_3392', 'gp_3396', 'gp_3399', 'gp_3401', 'gp_3403', 'gp_3405', 'gp_3409', 'gp_3411', 'gp_3415', 'gp_3420', 'gp_3422', 'gp_3423', 'gp_3427', 'gp_3428', 'gp_3436', 'gp_3442', 'gp_3445', 'gp_3447', 'gp_3449', 'gp_3452', 'gp_3453', 'gp_3454', 'gp_3456', 'gp_3459', 'gp_3461', 'gp_3463', 'gp_3464', 'gp_3465', 'gp_3466', 'gp_3468', 'gp_3470', 'gp_3472', 'gp_3473', 'gp_3474', 'gp_3476', 'gp_3478', 'gp_3480', 'gp_3481', 'gp_3482', 'gp_3484', 'gp_3486', 'gp_3487', 'gp_3488', 'gp_3489', 'gp_3490', 'gp_3492', 'gp_1729', 'gp_1744', 'gp_1745', 'gp_1747', 'gp_1749', 'gp_1750', 'gp_1752', 'gp_1753', 'gp_1755', 'gp_1756', 'gp_1757', 'gp_1759', 'gp_1760', 'gp_1762', 'gp_1763', 'gp_1765', 'gp_1767', 'gp_1768', 'gp_1770', 'gp_1771', 'gp_1772', 'gp_1773', 'gp_1774', 'gp_1775', 'gp_1778', 'gp_1781', 'gp_1784', 'gp_1786', 'gp_1788', 'gp_1790', 'gp_1791', 'gp_1793', 'gp_1795', 'gp_1796', 'gp_1798', 'gp_1803', 'gp_1804', 'gp_1806', 'gp_1807', 'gp_1810', 'gp_1813', 'gp_1815', 'gp_1818', 'gp_1822', 'gp_1823', 'gp_1825', 'gp_1826', 'gp_1828', 'gp_1829', 'gp_1832', 'gp_1833', 'gp_1835', 'gp_1836', 'gp_1838', 'gp_1839', 'gp_1841', 'gp_1842', 'gp_1844', 'gp_1848', 'gp_1849', 'gp_1852', 'gp_1853', 'gp_1856', 'gp_1857', 'gp_1861', 'gp_1862', 'gp_1863', 'gp_1864', 'gp_1867', 'gp_1869', 'gp_1872', 'gp_1874', 'gp_1875', 'gp_1879', 'gp_1882', 'gp_1883', 'gp_1886', 'gp_1888', 'gp_1891', 'gp_1892', 'gp_1893', 'gp_1895', 'gp_1896', 'gp_1900', 'gp_1904', 'gp_1905', 'gp_1906', 'gp_1910', 'gp_1911', 'gp_1912', 'gp_1914', 'gp_1918', 'gp_1922', 'gp_1925', 'gp_1928', 'gp_1934', 'gp_1936', 'gp_1939', 'gp_1948', 'gp_1949', 'gp_1952', 'gp_1953', 'gp_1954', 'gp_1955', 'gp_1959', 'gp_1960', 'gp_1963', 'gp_1965', 'gp_1968', 'gp_1972', 'gp_1974', 'gp_1977', 'gp_1980', 'gp_1982', 'gp_1984', 'gp_1989', 'gp_1990', 'gp_1991', 'gp_1992', 'gp_1994', 'gp_1996', 'gp_1999', 'gp_2000', 'gp_2003', 'gp_2004', 'gp_2007', 'gp_2009', 'gp_2010', 'gp_2011', 'gp_2012', 'gp_2014', 'gp_2015', 'gp_2016', 'gp_2017', 'gp_2021', 'gp_2022', 'gp_2027', 'gp_2028', 'gp_2031', 'gp_2033', 'gp_2034', 'gp_2035', 'gp_2036', 'gp_2038', 'gp_2041', 'gp_2042', 'gp_2044', 'gp_2046', 'gp_2048', 'gp_2052', 'gp_2054', 'gp_2055', 'gp_2056', 'gp_2057', 'gp_2059', 'gp_2062', 'gp_2063', 'gp_2065', 'gp_2067', 'gp_2069', 'gp_2071', 'gp_2072', 'gp_2073', 'gp_2074', 'gp_2075', 'gp_2085', 'gp_2086', 'gp_2088', 'gp_2090', 'gp_2092', 'gp_2093', 'gp_2094', 'gp_2097', 'gp_2099', 'gp_2101', 'gp_2103', 'gp_2105', 'gp_2106', 'gp_2108', 'gp_2110', 'gp_2111', 'gp_2112', 'gp_2116', 'gp_2118', 'gp_2124', 'gp_2126', 'gp_2127', 'gp_2129', 'gp_2132', 'gp_2135', 'gp_2136', 'gp_2138', 'gp_2144', 'gp_2153', 'gp_2162', 'gp_2164', 'gp_2166', 'gp_2173', 'gp_2176', 'gp_2177', 'gp_2181', 'gp_2184', 'gp_2186', 'gp_2188', 'gp_2191', 'gp_2192', 'gp_2200', 'gp_2203', 'gp_2205', 'gp_2209', 'gp_2212', 'gp_2213', 'gp_2216', 'gp_2218', 'gp_2223', 'gp_2224', 'gp_2225', 'gp_2227', 'gp_2228', 'gp_2230', 'gp_2231', 'gp_2234', 'gp_2236', 'gp_2238', 'gp_2240', 'gp_2241', 'gp_2243', 'gp_2244', 'gp_2246', 'gp_2250', 'gp_2253', 'gp_2254', 'gp_2256', 'gp_2258', 'gp_2261', 'gp_2262', 'gp_2264', 'gp_2266', 'gp_2271', 'gp_2274', 'gp_2277', 'gp_2278', 'gp_2281', 'gp_2285', 'gp_2289', 'gp_2295', 'gp_2296', 'gp_2300', 'gp_2302', 'gp_2305', 'gp_2308', 'gp_2309', 'gp_2313', 'gp_2314', 'gp_2315', 'gp_2317', 'gp_2319', 'gp_2322', 'gp_2326', 'gp_2328', 'gp_2329', 'gp_2331', 'gp_2334', 'gp_2335', 'gp_2336', 'gp_2337', 'gp_2338', 'gp_2345', 'gp_2346', 'gp_2347', 'gp_2351', 'gp_2352', 'gp_2353', 'gp_2354', 'gp_2355', 'gp_2356', 'gp_2357', 'gp_2358', 'gp_2359', 'gp_2361', 'gp_2362', 'gp_2364', 'gp_2366', 'gp_2367', 'gp_2370', 'gp_2372', 'gp_2374', 'gp_2376', 'gp_2380', 'gp_2382', 'gp_2385', 'gp_2386', 'gp_2388', 'gp_2390', 'gp_2393', 'gp_2395', 'gp_2396', 'gp_2397', 'gp_2399', 'gp_2403', 'gp_2404', 'gp_2406', 'gp_2408', 'gp_2410', 'gp_2414', 'gp_2417', 'gp_2419', 'gp_2422', 'gp_2425', 'gp_2426', 'gp_2429', 'gp_2432', 'gp_2435', 'gp_2437', 'gp_2438', 'gp_2440', 'gp_2441', 'gp_2442', 'gp_2444', 'gp_2448', 'gp_2450', 'gp_2451', 'gp_2455', 'gp_2461', 'gp_2462', 'gp_2463', 'gp_2464', 'gp_2466', 'gp_2467', 'gp_2470', 'gp_2471', 'gp_2472', 'gp_2474', 'gp_2483', 'gp_2492', 'gp_2494', 'gp_2496', 'gp_2500', 'gp_2502', 'gp_2503', 'gp_2504', 'gp_2506', 'gp_2507', 'gp_2508', 'gp_2509', 'gp_2511', 'gp_2512', 'gp_2513', 'gp_2514', 'gp_2515', 'gp_2517', 'gp_2518', 'gp_2519', 'gp_2531', 'gp_2533', 'gp_2535', 'gp_2536', 'gp_2539', 'gp_2543', 'gp_2544', 'gp_2545', 'gp_2548', 'gp_2552', 'gp_2555', 'gp_2556', 'gp_2557', 'gp_2559', 'gp_2560', 'gp_2561', 'gp_2563', 'gp_2564', 'gp_2565', 'gp_2566', 'gp_2567', 'gp_2568', 'gp_2569', 'gp_2570', 'gp_2571', 'gp_2572', 'gp_2573', 'gp_2574', 'gp_2575', 'gp_2577', 'gp_2578', 'gp_2579', 'gp_2581', 'gp_2588', 'gp_2591', 'gp_2592', 'gp_2595', 'gp_2597', 'gp_2599', 'gp_2603', 'gp_2604', 'gp_2606', 'gp_2608', 'gp_2609', 'gp_2611', 'gp_2613', 'gp_2617', 'gp_2619', 'gp_2621', 'gp_2623', 'gp_2625', 'gp_2629', 'gp_2631', 'gp_2632', 'gp_2635', 'gp_2637', 'gp_2641', 'gp_2642', 'gp_2643', 'gp_2647', 'gp_2649', 'gp_2650', 'gp_2652', 'gp_2654', 'gp_2655', 'gp_2660', 'gp_2661', 'gp_2665', 'gp_2666', 'gp_2668', 'gp_2671', 'gp_2672', 'gp_2675', 'gp_2677', 'gp_2681', 'gp_2682', 'gp_2684', 'gp_2685', 'gp_2697', 'gp_2699', 'gp_2701', 'gp_2702', 'gp_2705', 'gp_2706', 'gp_2708', 'gp_2711', 'gp_2716', 'gp_2720', 'gp_2721', 'gp_2722', 'gp_2724', 'gp_2726', 'gp_2728', 'gp_2730', 'gp_2733', 'gp_2736', 'gp_2738', 'gp_2741', 'gp_2744', 'gp_2750', 'gp_2753', 'gp_2756', 'gp_2758', 'gp_2760', 'gp_2761', 'gp_2766', 'gp_2768', 'gp_2773', 'gp_2776', 'gp_2779', 'gp_2780', 'gp_2782', 'gp_2784', 'gp_2785', 'gp_2789', 'gp_2797', 'gp_2800', 'gp_2807', 'gp_2811', 'gp_2814', 'gp_2817', 'gp_2820', 'gp_2822', 'gp_2823', 'gp_2827', 'gp_2828', 'gp_2837', 'gp_2838', 'gp_2839', 'gp_2859', 'gp_2864', 'gp_2868', 'gp_2870', 'gp_2880', 'gp_2884', 'gp_2885', 'gp_2892', 'gp_2895', 'gp_2896', 'gp_2902', 'gp_2908', 'gp_2910', 'gp_2918', 'gp_2919', 'gp_2922', 'gp_2924', 'gp_2926', 'gp_2937', 'gp_2940', 'gp_2942', 'gp_2944', 'gp_2945', 'gp_2958', 'gp_2961', 'gp_2963', 'gp_2966', 'gp_2968', 'gp_2970', 'gp_2971', 'gp_2976', 'gp_2981', 'gp_2986', 'gp_2989', 'gp_2994', 'gp_2997', 'gp_2999', 'gp_3000', 'gp_3018', 'gp_3024', 'gp_3025', 'gp_3030', 'gp_3033', 'gp_3036', 'gp_3041', 'gp_3044', 'gp_3046', 'gp_3057', 'gp_3066', 'gp_3068', 'gp_3072', 'gp_3075', 'gp_3078', 'gp_3080', 'gp_3083', 'gp_3084', 'gp_3091', 'gp_3097', 'gp_3098', 'gp_3100', 'gp_3101', 'gp_3103', 'gp_3104', 'gp_3108', 'gp_3118', 'gp_3121', 'gp_3122', 'gp_3123', 'gp_3124', 'gp_3129', 'gp_3130', 'gp_3133', 'gp_3136', 'gp_3138', 'gp_3139', 'gp_3143', 'gp_3147', 'gp_3152', 'gp_3154', 'gp_3159', 'gp_3161', 'gp_3169', 'gp_3173', 'gp_3176', 'gp_3177', 'gp_3178', 'gp_3181', 'gp_3183', 'gp_3184', 'gp_3187', 'gp_3189', 'gp_3191', 'gp_3192', 'gp_3194', 'gp_3197', 'gp_3201', 'gp_3208', 'gp_3210', 'gp_3213', 'gp_3215', 'gp_3218', 'gp_3219', 'gp_3220', 'gp_3230', 'gp_3232', 'gp_3235', 'gp_3237', 'gp_3241', 'gp_3243', 'gp_3281', 'gp_2653', 'gp_2657', 'gp_2663', 'gp_2664', 'gp_2669', 'gp_2673', 'gp_2674', 'gp_2676', 'gp_2678', 'gp_2680', 'gp_2689', 'gp_2690', 'gp_2691', 'gp_2692', 'gp_2693', 'gp_2694', 'gp_2695', 'gp_2696', 'gp_2698', 'gp_2712', 'gp_2713', 'gp_2714', 'gp_2715', 'gp_2717', 'gp_2718', 'gp_2723', 'gp_2727', 'gp_2731', 'gp_2734', 'gp_2739', 'gp_2745', 'gp_2748', 'gp_2754', 'gp_2755', 'gp_2757', 'gp_2759', 'gp_2765', 'gp_2769', 'gp_2775', 'gp_2778', 'gp_2793', 'gp_2794', 'gp_2795', 'gp_2796', 'gp_2798', 'gp_2801', 'gp_2803', 'gp_2806', 'gp_2809', 'gp_2810', 'gp_2813', 'gp_2815', 'gp_2824', 'gp_2826', 'gp_2829', 'gp_2831', 'gp_2836', 'gp_2840', 'gp_2841', 'gp_2843', 'gp_2844', 'gp_2848', 'gp_2853', 'gp_2855', 'gp_2856', 'gp_2858', 'gp_2860', 'gp_2863', 'gp_2865', 'gp_2869', 'gp_2874', 'gp_2883', 'gp_2886', 'gp_2887', 'gp_2890', 'gp_2893', 'gp_2909', 'gp_2911', 'gp_2912', 'gp_2920', 'gp_2921', 'gp_2923', 'gp_2925', 'gp_2951', 'gp_2954', 'gp_2974', 'gp_2977', 'gp_2979', 'gp_2984', 'gp_2985', 'gp_2993', 'gp_2995', 'gp_2996', 'gp_3002', 'gp_3003', 'gp_3008', 'gp_3010', 'gp_3014', 'gp_3015', 'gp_3017', 'gp_3023', 'gp_3026', 'gp_3031', 'gp_3034', 'gp_3037', 'gp_3042', 'gp_3043', 'gp_3049', 'gp_3052', 'gp_3056', 'gp_3058', 'gp_3060', 'gp_3063', 'gp_3069', 'gp_3074', 'gp_3077', 'gp_3082', 'gp_3095', 'gp_3096', 'gp_3099', 'gp_3102', 'gp_3105', 'gp_3109', 'gp_3110', 'gp_3113', 'gp_3116', 'gp_3117', 'gp_3119', 'gp_3124', 'gp_3125', 'gp_3128', 'gp_3131', 'gp_3132', 'gp_3134', 'gp_3137', 'gp_3141', 'gp_3143', 'gp_3144', 'gp_3146', 'gp_3148', 'gp_3151', 'gp_3153', 'gp_3155', 'gp_3158', 'gp_3162', 'gp_3163', 'gp_3166', 'gp_3167', 'gp_3172', 'gp_3175', 'gp_3180', 'gp_3182', 'gp_3185', 'gp_3188', 'gp_3190', 'gp_3193', 'gp_3196', 'gp_3206', 'gp_3207', 'gp_3211', 'gp_3213', 'gp_3219', 'gp_3221', 'gp_3223', 'gp_3225', 'gp_3227', 'gp_3228', 'gp_3234', 'gp_3236', 'gp_3240', 'gp_3242', 'gp_3244', 'gp_3247', 'gp_3248', 'gp_3250', 'gp_3253', 'gp_3255', 'gp_3259', 'gp_3261', 'gp_3265', 'gp_3274', 'gp_3275', 'gp_3292', 'gp_3293', 'gp_3295', 'gp_3298', 'gp_3301', 'gp_3303', 'gp_3308', 'gp_3312', 'gp_3314', 'gp_3317', 'gp_3321', 'gp_3324', 'gp_3327', 'gp_3328', 'gp_3331', 'gp_3332', 'gp_3333', 'gp_3340', 'gp_3345', 'gp_3346', 'gp_3355', 'gp_3357', 'gp_3360', 'gp_3366', 'gp_3371', 'gp_3376', 'gp_3378', 'gp_3383', 'gp_3389', 'gp_3394', 'gp_3397', 'gp_3400', 'gp_3404', 'gp_3408', 'gp_3410', 'gp_3414', 'gp_3416', 'gp_3429', 'gp_3430', 'gp_3431', 'gp_3433', 'gp_3434', 'gp_3437', 'gp_3439', 'gp_3443', 'gp_3444', 'gp_3446', 'gp_3448', 'gp_3450', 'gp_3451', 'gp_3455', 'gp_3457', 'gp_3458', 'gp_3460', 'gp_3462', 'gp_3467', 'gp_3469', 'gp_3471', 'gp_3474', 'gp_3477', 'gp_3479', 'gp_3483', 'gp_3485', 'gp_3491', 'gp_3493', 'gp_3494', 'gp_3495', 'gp_3496', 'gp_3497', 'gp_3499', 'gp_3500', 'gp_3502', 'gp_3516', 'gp_3518', 'gp_3523', 'gp_3524', 'gp_3525', 'gp_3526', 'gp_3528', 'gp_3529', 'gp_3530', 'gp_3539', 'gp_3541', 'gp_3542', 'gp_3545', 'gp_3548', 'gp_3551', 'gp_3553', 'gp_3555', 'gp_3558', 'gp_3564', 'gp_3567', 'gp_3572', 'gp_3575', 'gp_3576', 'gp_3578', 'gp_3580', 'gp_3588', 'gp_3589', 'gp_3591', 'gp_3598', 'gp_3606', 'gp_3608', 'gp_3610', 'gp_3614', 'gp_3617', 'gp_3621', 'gp_4134', 'gp_4139', 'gp_6938', 'gp_6939', 'gp_6940', 'gp_6941', 'gp_6942', 'gp_6946', 'gp_6972', 'gp_6973', 'gp_6974', 'gp_6975', 'gp_6976', 'gp_6977', 'gp_6978', 'gp_6979', 'gp_1285', 'gp_1286', 'gp_1287', 'gp_1290', 'gp_1294', 'gp_1298', 'gp_1300', 'gp_1302', 'gp_1306', 'gp_1309', 'gp_1311', 'gp_1321', 'gp_1327', 'gp_1328', 'gp_1334', 'gp_1338', 'gp_1344', 'gp_1347', 'gp_1353', 'gp_1355', 'gp_1357', 'gp_1361', 'gp_1367', 'gp_1370', 'gp_1371', 'gp_1374', 'gp_1379', 'gp_1384', 'gp_1387', 'gp_1391', 'gp_1394', 'gp_1397', 'gp_1401', 'gp_1403', 'gp_1405', 'gp_1410', 'gp_1415', 'gp_1418', 'gp_1419', 'gp_1423', 'gp_1426', 'gp_1431', 'gp_1434', 'gp_1437', 'gp_1440', 'gp_1444', 'gp_1446', 'gp_1450', 'gp_1453', 'gp_1457', 'gp_1459', 'gp_1461', 'gp_1464', 'gp_1465', 'gp_1467', 'gp_1468', 'gp_1469', 'gp_1470', 'gp_1471', 'gp_1472', 'gp_1473', 'gp_1475', 'gp_1479', 'gp_1480', 'gp_1481', 'gp_1487', 'gp_1490', 'gp_1494', 'gp_1495', 'gp_1499', 'gp_1502', 'gp_1505', 'gp_1507', 'gp_1511', 'gp_1514', 'gp_1517', 'gp_1520', 'gp_1525', 'gp_1528', 'gp_1530', 'gp_1536', 'gp_1538', 'gp_1540', 'gp_1542', 'gp_1545', 'gp_1546', 'gp_1548', 'gp_1552', 'gp_1554', 'gp_1556', 'gp_1558', 'gp_1559', 'gp_1560', 'gp_1563', 'gp_1564', 'gp_1565', 'gp_1566', 'gp_1568', 'gp_1570', 'gp_1572', 'gp_1573', 'gp_1575', 'gp_1576', 'gp_1580', 'gp_1582', 'gp_1583', 'gp_1584', 'gp_1585', 'gp_1588', 'gp_1589', 'gp_1592', 'gp_1594', 'gp_1596', 'gp_1598', 'gp_1600', 'gp_1601', 'gp_1604', 'gp_1608', 'gp_1609', 'gp_1610', 'gp_1612', 'gp_1617', 'gp_1619', 'gp_1620', 'gp_1621', 'gp_1624', 'gp_1626', 'gp_1630', 'gp_1632', 'gp_1634', 'gp_1635', 'gp_1637', 'gp_1640', 'gp_1642', 'gp_1643', 'gp_1645', 'gp_1647', 'gp_1651', 'gp_1652', 'gp_1656', 'gp_1657', 'gp_1659', 'gp_1661', 'gp_1663', 'gp_1665', 'gp_1666', 'gp_1680', 'gp_1685', 'gp_1686', 'gp_1688', 'gp_1690', 'gp_1693', 'gp_1695', 'gp_1696', 'gp_1700', 'gp_1701', 'gp_1702', 'gp_1704', 'gp_1705', 'gp_1707', 'gp_1709', 'gp_1715', 'gp_1717', 'gp_1718', 'gp_1720', 'gp_1721', 'gp_1722', 'gp_1723', 'gp_1724', 'gp_1725', 'gp_1726', 'gp_4154', 'gp_4155', 'gp_4156', 'gp_4160', 'gp_4161', 'gp_4163', 'gp_4164', 'gp_4165', 'gp_4166', 'gp_4170', 'gp_4172', 'gp_4173', 'gp_4182', 'gp_4183', 'gp_4184', 'gp_4185', 'gp_4186', 'gp_4187', 'gp_4188', 'gp_4189', 'gp_4190', 'gp_4191', 'gp_4192', 'gp_4193', 'gp_4194', 'gp_4195', 'gp_4196', 'gp_4197', 'gp_4198', 'gp_4199', 'gp_4200', 'gp_4201', 'gp_4202', 'gp_4204', 'gp_4205', 'gp_4209', 'gp_4211', 'gp_4213', 'gp_4214', 'gp_4215', 'gp_4216', 'gp_4218', 'gp_4219', 'gp_4220', 'gp_4221', 'gp_4222', 'gp_4225', 'gp_4226', 'gp_4006', 'gp_4012', 'gp_4013', 'gp_4016', 'gp_4018', 'gp_4019', 'gp_4022', 'gp_4023', 'gp_4024', 'gp_4025', 'gp_4026', 'gp_4027', 'gp_4028', 'gp_4029', 'gp_4032', 'gp_4033', 'gp_4034', 'gp_4037', 'gp_4038', 'gp_4041', 'gp_4042', 'gp_4044', 'gp_4045', 'gp_4046', 'gp_4050', 'gp_4054', 'gp_4072', 'gp_4073', 'gp_4074', 'gp_4075', 'gp_4077', 'gp_4080', 'gp_4081', 'gp_4082', 'gp_4083', 'gp_4084', 'gp_4122', 'gp_4123', 'gp_4124', 'gp_5744', 'gp_5771', 'gp_5772', 'gp_5776', 'gp_5780', 'gp_5785', 'gp_5788', 'gp_5792', 'gp_5801', 'gp_5808', 'gp_5813', 'gp_5814', 'gp_5815', 'gp_5816', 'gp_5817', 'gp_5818', 'gp_5819', 'gp_5820', 'gp_5822', 'gp_5824', 'gp_5825', 'gp_5826', 'gp_5827', 'gp_5828', 'gp_5829', 'gp_5830', 'gp_5831', 'gp_5832', 'gp_5833', 'gp_5834', 'gp_5835', 'gp_5836', 'gp_5837', 'gp_5839', 'gp_5840', 'gp_5841', 'gp_5842', 'gp_5843', 'gp_5844', 'gp_5845', 'gp_5846', 'gp_5847', 'gp_5848', 'gp_5849', 'gp_5850', 'gp_5851', 'gp_5852', 'gp_5853', 'gp_5854', 'gp_5855', 'gp_5856', 'gp_5857', 'gp_5859', 'gp_5861', 'gp_5862', 'gp_5865', 'gp_5867', 'gp_5869', 'gp_5871', 'gp_5872', 'gp_5873', 'gp_5875', 'gp_5876', 'gp_5877', 'gp_5878', 'gp_5879', 'gp_5880', 'gp_5881', 'gp_5882', 'gp_5883', 'gp_5884', 'gp_5885', 'gp_5887', 'gp_5889', 'gp_5894', 'gp_5900', 'gp_5991', 'gp_6800', 'gp_6807', 'gp_5752', 'gp_5753', 'gp_5756', 'gp_5757', 'gp_5760', 'gp_5761', 'gp_5764', 'gp_5769', 'gp_5770', 'gp_5774', 'gp_5778', 'gp_5783', 'gp_5789', 'gp_5794', 'gp_5803', 'gp_5809', 'gp_5810', 'gp_5811', 'gp_5812', 'gp_5907', 'gp_5908', 'gp_5909', 'gp_5911', 'gp_5913', 'gp_5915', 'gp_5918', 'gp_5924', 'gp_5925', 'gp_5931', 'gp_5935', 'gp_5937', 'gp_5951', 'gp_5971', 'gp_5973', 'gp_5978', 'gp_5986', 'gp_5987', 'gp_5990', 'gp_5992', 'gp_5994', 'gp_5995', 'gp_5996', 'gp_5997', 'gp_5998', 'gp_6001', 'gp_6002', 'gp_6003', 'gp_6005', 'gp_6007', 'gp_6010', 'gp_6011', 'gp_6015', 'gp_6016', 'gp_6018', 'gp_6019', 'gp_6022', 'gp_6023', 'gp_6510', 'gp_6511', 'gp_6512', 'gp_6513', 'gp_6516', 'gp_6517', 'gp_6518', 'gp_6519', 'gp_6520', 'gp_6521', 'gp_6522', 'gp_6523', 'gp_6524', 'gp_6525', 'gp_6527', 'gp_6528', 'gp_6529', 'gp_6531', 'gp_6603', 'gp_6604', 'gp_6606', 'gp_6607', 'gp_6608', 'gp_6609', 'gp_6610', 'gp_5163', 'gp_6035', 'gp_6037', 'gp_6040', 'gp_6041', 'gp_6043', 'gp_6051', 'gp_6056', 'gp_6058', 'gp_6059', 'gp_6062', 'gp_6063', 'gp_6065', 'gp_6067', 'gp_6070', 'gp_6081', 'gp_6083', 'gp_6085', 'gp_6088', 'gp_6091', 'gp_6092', 'gp_6094', 'gp_6095', 'gp_6096', 'gp_6102', 'gp_6103', 'gp_6107', 'gp_6108', 'gp_6111', 'gp_6112', 'gp_6114', 'gp_6115', 'gp_6117', 'gp_6119', 'gp_6120', 'gp_6121', 'gp_6122', 'gp_6123', 'gp_6124', 'gp_6125', 'gp_6127', 'gp_6129', 'gp_6130', 'gp_6131', 'gp_6132', 'gp_6133', 'gp_6136', 'gp_6137', 'gp_6138', 'gp_6139', 'gp_6140', 'gp_6141', 'gp_6142', 'gp_6143', 'gp_6144', 'gp_6145', 'gp_6147', 'gp_6148', 'gp_6149', 'gp_6150', 'gp_6152', 'gp_6158', 'gp_6159', 'gp_6162', 'gp_6165', 'gp_6166', 'gp_6167', 'gp_6168', 'gp_6169', 'gp_6171', 'gp_6173', 'gp_6176', 'gp_6178', 'gp_6179', 'gp_6180', 'gp_6185', 'gp_6188', 'gp_6189', 'gp_6190', 'gp_6193', 'gp_6195', 'gp_6196']
        except: 
            pass

        interacting_peds = ['gp_3308', 'gp_5852', 'gp_4220', 'gp_2920', 'gp_2924', 'gp_3074', 'gp_3043', 'gp_2578', 'gp_2579', 'gp_6158', 'gp_6138', 'gp_3471', 'gp_2592', 'gp_2908', 'gp_3162', 'gp_2597', 'gp_3013', 'gp_2094', 'gp_2097', 'gp_2609', 'gp_2101', 'gp_3545', 'gp_2105', 'gp_3469', 'gp_3077', 'gp_2638', 'gp_3529', 'gp_2129.0', 'gp_2723', 'gp_2138', 'gp_2144', 'gp_3056', 'gp_4190', 'gp_3036', 'gp_6137', 'gp_3564', 'gp_3193', 'gp_2997', 'gp_2682', 'gp_2968', 'gp_5971', 'gp_3274', 'gp_6946', 'gp_6877', 'gp_5748', 'gp_2724', 'gp_5750', 'gp_3575', 'gp_3082', 'gp_3606', 'gp_2744', 'gp_6979', 'gp_3321', 'gp_3614', 'gp_2778', 'gp_1419', 'gp_5931', 'gp_3567', 'gp_3105', 'gp_3060', 'gp_3467', 'gp_4027', 'gp_4039', 'gp_3221', 'gp_4192', 'gp_2645', 'gp_3326', 'gp_3509', 'gp_3394', 'gp_6517', 'gp_3334', 'gp_6942', 'gp_3558', 'gp_2837', 'gp_2838', 'gp_6975', 'gp_3058', 'gp_3530', 'gp_6978', 'gp_5789', 'gp_3503', 'gp_2870', 'gp_2880', 'gp_4191', 'gp_6938', 'gp_3109', 'gp_3555', 'gp_3576', 'gp_2644', 'gp_2892', 'gp_2895', 'gp_6159', 'gp_3477', 'gp_2775', 'gp_2727', 'gp_2926', 'gp_3247', 'gp_3491', 'gp_2967', 'gp_6945', 'gp_6136', 'gp_3095', 'gp_2981', 'gp_2425', 'gp_2426', 'gp_2429', 'gp_4043', 'gp_3052', 'gp_4016', 'gp_2444', 'gp_2448', 'gp_2977', 'gp_2450', 'gp_5809', 'gp_4199', 'gp_1955', 'gp_3474', 'gp_5794', 'gp_2883', 'gp_2765', 'gp_2869', 'gp_2910', 'gp_3598', 'gp_3572', 'gp_5851', 'gp_6976', 'gp_6972', 'gp_5751', 'gp_3548', 'gp_4040', 'gp_2504', 'gp_1302', 'gp_2524', 'gp_2526', 'gp_3526', 'gp_6973', 'gp_3479', 'gp_3099', 'gp_5803', 'gp_2694', 'gp_2545', 'gp_2548', 'gp_4045', 'gp_3034', 'gp_1306']

        # For type of signalized

        # For day night
        try:
            # Night time videos
            if params['time'] == 'night' and image_set == 'test':
                vid_list = ['gp_set_0003_vid_0005','gp_set_0005_vid_0005','gp_set_0008_vid_0001','gp_set_0008_vid_0004','gp_set_0008_vid_0005']

            # Day time videos
            if params['time'] == 'day' and image_set == 'test':
                print("I AM HERE FOR GOD'S SAKE")
                vid_list = ['gp_set_0003_vid_0001','gp_set_0003_vid_0002','gp_set_0003_vid_0003','gp_set_0003_vid_0004','gp_set_0005_vid_0001',\
                            'gp_set_0005_vid_0002','gp_set_0005_vid_0003','gp_set_0009_vid_0001']
        except:
            pass

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                if vid not in vid_list:
                    continue
                img_width = annotations[sid][vid]['width']
                img_height = annotations[sid][vid]['height']
                pid_annots = annotations[sid][vid]['pedestrian_annotations']
                # ############# For debugging ##################
                # random.seed(2)
                # pid_random = random.choice(list(pid_annots.keys())) # Just for debugging
                # pid_annots = {pid_random: pid_annots[pid_random]}
                # # print(pid_annots)
                # ############# End of debugging ################

                vid_annots = annotations[sid][vid]['vehicle_annotations'] 
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    if pid not in all_peds:
                        continue

                    try:
                        if params['interaction'] == 'y' and pid not in interacting_peds: # We only want interacting folks
                            continue
                        if params['interaction'] == 'n' and pid in interacting_peds:
                            continue
                    except:
                        pass
                    num_pedestrians += 1
                    frame_ids = pid_annots[pid]['frames']
                    boxes = pid_annots[pid]['bbox']
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion']

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)

                    if len(boxes) / seq_stride < params['min_track_size']:
                        continue

                    # For occlusion vs no-occlusion
                    occlusions_imp = occlusions[:]
                    occlusion_fraction = (occlusions_imp.count(1) + occlusions_imp.count(2))/len(occlusions_imp)
                    # print(occlusion_fraction)
                    try:
                        if params['occluded'] == 'y' and occlusion_fraction < 0.3:
                            # print("one")
                            continue # Don't consider that pedestrian since we are only considering occluded pedestrians
                        if params['occluded'] == 'n' and occlusion_fraction > 0.3:
                            # print("two")
                            continue # Don't consider that pedestrian since we are only considering non-occluded pedestrians
                        # pdb.set_trace()
                    except:
                        pass

                    if sq_ratio:
                        boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                    try:
                        intent = [[pid_annots[pid]['attributes']['crossing_motive']]] * len(boxes)
                    except KeyError:
                        continue # No key "attributes" present in the dictionary, so skip this pedestrian.

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    resolutions = [[img_width, img_height]] * len(boxes)
                    resolution_seq.append(resolutions[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent_seq.append(intent[::seq_stride])

                    # accT_seq.append([[0.0] for i in frame_ids][::seq_stride])
                    # obds_seq.append([[0.0] for i in frame_ids][::seq_stride])
                    # accX_seq.append([[0.0] for i in frame_ids][::seq_stride])
                    # accY_seq.append([[0.0] for i in frame_ids][::seq_stride])
                    # accZ_seq.append([[0.0] for i in frame_ids][::seq_stride])

                    accT_seq.append([[vid_annots[i]['accT']] for i in frame_ids][::seq_stride])
                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                    accX_seq.append([[vid_annots[i]['accX']] for i in frame_ids][::seq_stride])
                    accY_seq.append([[vid_annots[i]['accY']] for i in frame_ids][::seq_stride])
                    accZ_seq.append([[vid_annots[i]['accZ']] for i in frame_ids][::seq_stride])
                    

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'resolution': resolution_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'accT': accT_seq,
                'accX': accX_seq,
                'accY': accY_seq,
                'accZ': accZ_seq,
                'intention_prob': intent_seq}

    def _get_post_crossing(self, image_set, annotations, **params):
        """
        Generates crossing data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generate_database)
        :return: A dictionary of crossing data.
        """

        print('---------------------------------------------------------')
        print("Generating sequence data post crossing")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        obds_seq = []
        activities = []
        group_sizes = []
        accT_seq, accX_seq, accY_seq, accZ_seq = [], [], [], []
        image_dimensions_seq = []
        
        
        set_ids, _pids = self._get_data_ids(image_set, params)
        
        # if len(set_ids) != 5: # Only equal when data_split_type is not default.
        #     print(len(set_ids))
        #     print("Considering the sets",set_ids,"for",image_set)
        # else:
        #     category = image_set.split('_')[1]
        #     print(f"Considering pedestrians of category - {category} for testing.")
        
        
        if '_' in image_set:
            # Now that we've gotten the ped_ids we want, we can make image_set one of 'train', 'val', 'test'.
            image_set = image_set.split('_')[0]

        print("====================================")
        print("For {},".format(image_set),end=" ")
        print("Considering the tracks post the crossing point as candidates")



        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                img_height = annotations[sid][vid]['height']
                image_dimensions_seq.append((img_width,img_height))
                pid_annots = annotations[sid][vid]['pedestrian_annotations']
                vid_annots = annotations[sid][vid]['vehicle_annotations']
                for pid in sorted(pid_annots):
                    # if params['data_split_type'] != 'default' and pid not in _pids:
                    #     continue
                    num_pedestrians += 1
                    try:
                        _ = pid_annots[pid]['attributes']
                    except:
                        print("Pedestrian {} doesn't have corresponding attributes. Skipping".format(pid))
                        continue
                    frame_ids = pid_annots[pid]['frames'] # All frames that the pedestrian is in.
                    event_frame = pid_annots[pid]['attributes']['crossing_point']
                    
                    # Check if the event frame that we got above lies within a pedestrian track.
                    try:
                        start_idx = frame_ids.index(event_frame)
                    except ValueError:
                        # print(f"The frame {event_frame} not in range {frame_ids[0],frame_ids[-1]} for {pid}. Recheck annotations. Skipping.")
                        continue
                    
                    # From here on, we consider the frames post the event frame.
                    boxes = pid_annots[pid]['bbox'][start_idx:] 
                    frame_ids = frame_ids[start_idx:]
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids] # Get the list of image paths for the frames upto the event frame.
                    occlusions = pid_annots[pid]['occlusion'][start_idx:] # Get the list of occlusions for the frames upto the event frame.

                    # Get only the frame ids for which the height of the bboxes is in the
                    # specified range.
                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)
                        
                    # The resulting lists from above are such that the elements corresponding to bounding boxes which are smaller than a 
                    # height are omitted. This is done to ensure that the data is clean and only the relevant data is considered.

                    if len(boxes) / seq_stride < params['min_track_size']:
                        print("Skipping track of pedestrian",pid)
                        continue

                    if sq_ratio:
                        boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                    # The seq_stride and overlap param are different. seq_stride specifies how many frames you
                    # want to skip while considering each sequence.  

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent = [[pid_annots[pid]['attributes']['crossing_motive']]] * len(boxes)
                    intent_seq.append(intent[::seq_stride])

                    acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)
                    activities.append(acts[::seq_stride])

                    grps = [[pid_annots[pid]['attributes']['group_size']]] * len(boxes)
                    group_sizes.append(grps[::seq_stride])


                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                    accT_seq.append([[vid_annots[i]['accT']] for i in frame_ids][::seq_stride])
                    accX_seq.append([[vid_annots[i]['accX']] for i in frame_ids][::seq_stride])
                    accY_seq.append([[vid_annots[i]['accY']] for i in frame_ids][::seq_stride])
                    accZ_seq.append([[vid_annots[i]['accZ']] for i in frame_ids][::seq_stride])
                    

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        # Every element in the dictionary is a list of lists. Each list corresponds to a pedestrian track of interest. Except for image_dimension_seq, which is a list of tuples.
        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'accT': accT_seq,
                'accX': accX_seq,
                'accY': accY_seq,
                'accZ': accZ_seq,
                'intention_vals': intent_seq,
                'activities': activities,
                'group': group_sizes,
                'image_dimension': image_dimensions_seq}


    # def _get_crossing(self, image_set, annotations, **params):
    #     """
    #     Generates crossing data.
    #     :param image_set: Data split to use
    #     :param annotations: Annotations database
    #     :param params: Parameters to generate data (see generate_database)
    #     :return: A dictionary of crossing data.
    #     """

    #     print('---------------------------------------------------------')
    #     print("Generating crossing data")

    #     num_pedestrians = 0
    #     seq_stride = params['fstride']
    #     sq_ratio = params['squarify_ratio']
    #     height_rng = params['height_rng']

    #     image_seq, pids_seq = [], []
    #     box_seq, center_seq, occ_seq = [], [], []
    #     intent_seq = []
    #     obds_seq = []
    #     activities = []
    #     group_sizes = []
    #     accT_seq, accX_seq, accY_seq, accZ_seq = [], [], [], []
    #     image_dimensions_seq = []
        
        
    #     set_ids, _pids = self._get_data_ids(image_set, params)
        
    #     if len(set_ids) != 5: # Only equal when data_split_type is not default.
    #         print(len(set_ids))
    #         print("Considering the sets",set_ids,"for",image_set)
    #     else:
    #         category = image_set.split('_')[1]
    #         print(f"Considering pedestrians of category - {category} for testing.")

    #     if '_' in image_set:
    #         # Now that we've gotten the ped_ids we want, we can make image_set one of 'train', 'val', 'test'.
    #         image_set = image_set.split('_')[0]

    #     print("====================================")
    #     print(f"For {image_set},",end=" ")
    #     if params[f'{image_set}_seq_end'] == 'crossing_point':
    #         print("considering the tracks upto the crossing point and before the TTE as candidates")
    #     elif params[f'{image_set}_seq_end'] == 'track_end':
    #         print("considering the entire track before the TTE as candidates.") 
    #     print("====================================")



    #     for sid in set_ids:
    #         for vid in sorted(annotations[sid]):
    #             img_width = annotations[sid][vid]['width']
    #             img_height = annotations[sid][vid]['height']
    #             image_dimensions_seq.append((img_width,img_height))
    #             pid_annots = annotations[sid][vid]['pedestrian_annotations']
    #             vid_annots = annotations[sid][vid]['vehicle_annotations']
    #             for pid in sorted(pid_annots):
    #                 if params['data_split_type'] != 'default' and pid not in _pids:
    #                     continue
    #                 num_pedestrians += 1

    #                 frame_ids = pid_annots[pid]['frames'] # All frames that the pedestrian is in.
    #                 if params[f'{image_set}_seq_end'] == 'crossing_point':
    #                     try:
    #                         event_frame = pid_annots[pid]['attributes']['crossing_point']
    #                     except KeyError:
    #                         print(f"Pedestrian {pid} doesn't have corresponding attributes. Skipping")
    #                         continue
    #                 elif params[f'{image_set}_seq_end'] == 'track_end':
    #                     try:
    #                         event_frame = frame_ids[-1]
    #                         crossing_point = pid_annots[pid]['attributes']['crossing_point']
    #                     except KeyError:
    #                         print(f"Pedestrian {pid} doesn't have corresponding attributes. Skipping")
    #                         continue
                    
    #                 # Check if the event frame that we got above lies within a pedestrian track.
    #                 try:
    #                     end_idx = frame_ids.index(event_frame)
    #                 except ValueError:
    #                     print(f"The frame {event_frame} not in range {frame_ids[0],frame_ids[-1]} for {pid}. Recheck annotations. Skipping.")
    #                     continue
                    
    #                 # From here on, we consider the frames only upto the event frame, depending on the seq_end.
    #                 boxes = pid_annots[pid]['bbox'][:end_idx + 1] 
    #                 frame_ids = frame_ids[:end_idx + 1]
    #                 images = [self._get_image_path(sid, vid, f) for f in frame_ids] # Get the list of image paths for the frames upto the event frame.
    #                 occlusions = pid_annots[pid]['occlusion'][:end_idx + 1] # Get the list of occlusions for the frames upto the event frame.

    #                 # Get only the frame ids for which the height of the bboxes is in the
    #                 # specified range.
    #                 if height_rng[0] > 0 or height_rng[1] < float('inf'):
    #                     images, boxes, frame_ids, occlusions = self._height_check(height_rng,
    #                                                                               frame_ids, boxes,
    #                                                                               images, occlusions)
                        
    #                 # The resulting lists from above are such that the elements corresponding to bounding boxes which are smaller than a 
    #                 # height are omitted. This is done to ensure that the data is clean and only the relevant data is considered.

    #                 if len(boxes) / seq_stride < params['min_track_size']:
    #                     print("Skipping track of pedestrian",pid)
    #                     continue

    #                 if sq_ratio:
    #                     boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

    #                 # The seq_stride and overlap param are different. seq_stride specifies how many frames you
    #                 # want to skip while considering each sequence.  

    #                 image_seq.append(images[::seq_stride])
    #                 box_seq.append(boxes[::seq_stride])
    #                 center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
    #                 occ_seq.append(occlusions[::seq_stride])

    #                 ped_ids = [[pid]] * len(boxes)
    #                 pids_seq.append(ped_ids[::seq_stride])

    #                 intent = [[pid_annots[pid]['attributes']['crossing_motive']]] * len(boxes)
    #                 intent_seq.append(intent[::seq_stride])

    #                 acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)
    #                 activities.append(acts[::seq_stride])

    #                 grps = [[pid_annots[pid]['attributes']['group_size']]] * len(boxes)
    #                 group_sizes.append(grps[::seq_stride])

    #                 obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
    #                 accT_seq.append([[vid_annots[i]['accT']] for i in frame_ids][::seq_stride])
    #                 accX_seq.append([[vid_annots[i]['accX']] for i in frame_ids][::seq_stride])
    #                 accY_seq.append([[vid_annots[i]['accY']] for i in frame_ids][::seq_stride])
    #                 accZ_seq.append([[vid_annots[i]['accZ']] for i in frame_ids][::seq_stride])
                    

    #     print('Subset: %s' % image_set)
    #     print('Number of pedestrians: %d ' % num_pedestrians)
    #     print('Total number of samples: %d ' % len(image_seq))

    #     # Every element in the dictionary is a list of lists. Each list corresponds to a pedestrian track of interest. Except for image_dimension_seq, which is a list of tuples.
    #     return {'image': image_seq,
    #             'pid': pids_seq,
    #             'bbox': box_seq,
    #             'center': center_seq,
    #             'occlusion': occ_seq,
    #             'obd_speed': obds_seq,
    #             'accT': accT_seq,
    #             'accX': accX_seq,
    #             'accY': accY_seq,
    #             'accZ': accZ_seq,
    #             'intention_vals': intent_seq,
    #             'activities': activities,
    #             'group': group_sizes,
    #             'image_dimension': image_dimensions_seq}

    def _get_intention(self, image_set, annotations, **params):
        """
        Generates intention data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """
        print('---------------------------------------------------------')
        print("Generating intention data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        intention_prob, intention_binary = [], []
        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['pedestrian_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1
                    # exp_start_frame = pid_annots[pid]['attributes']['exp_start_point']
                    try:
                        critical_frame = pid_annots[pid]['attributes']['crossing_point'] # changed critical_point to crossing_point.
                    except:
                        continue
                    frames = pid_annots[pid]['frames']

                    # start_idx = frames.index(exp_start_frame)
                    try:
                        end_idx = frames.index(critical_frame)
                    except:
                        continue
                    start_idx = max(0,end_idx - 60) # 60 frames (~2 seconds prior to crossing_point or the starting point, whichever is last.)

                    boxes = pid_annots[pid]['bbox'][start_idx:end_idx + 1]
                    frame_ids = frames[start_idx:end_idx + 1]
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion'][start_idx:end_idx + 1]

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)
                        
                    # print("The min track size is",params['min_track_size']) min_track_size was 0.
                    if len(boxes) / seq_stride < params['min_track_size']:
                        continue

                    if sq_ratio:
                        boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                    int_prob = [[pid_annots[pid]['attributes']['crossing_motive']]] * len(boxes)
                    int_bin = [[int(pid_annots[pid]['attributes']['crossing_motive'] >= 0.5)]] * len(boxes) # so "maybe" is considered 1.

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    intention_prob.append(int_prob[::seq_stride])
                    intention_binary.append(int_bin[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        # for i in box_seq:
        #     if len(i) < 15:
        #         print("Length of bbox is",len(i))

        return {'image': image_seq,
                'bbox': box_seq,
                'occlusion': occ_seq,
                'intention_prob': intention_prob,
                'intention_binary': intention_binary,
                'ped_id': pids_seq}

    def _get_behaviors(self,height_rng=[100, float('inf')],seq_stride=1):
        print('---------------------------------------------------------')
        print("Generating behavior data")
        print('---------------------------------------------------------')
        annotations = self.generate_database()
        num_pedestrians = 0

        image_seq = []
        box_seq, beh_label_seq = [], []
        # set_ids, _pids = self._get_data_ids(image_set, params)
        # print("Considering the sets",set_ids) # Only if the data_split_type is default.   
        set_ids = [f for f in sorted(listdir(join(self._images_path,"gopro")))]
        print("Collecting Behavioral labels...")
        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                if vid in ['gp_set_0002_vid_0003','gp_set_0005_vid_0003','gp_set_0002_vid_0004','gp_set_0005_vid_0002','gp_set_0006_vid_0002']: # Vids to exclude from behavioral annotations as of now, since they don't have beh attributes properly given.
                    print("Skipping vid",vid,"from behavioral data.")
                    continue
                pid_annots = annotations[sid][vid]['pedestrian_annotations']
                for pid in sorted(pid_annots):
                    num_pedestrians += 1
                    frame_ids = pid_annots[pid]['frames'] # All frames that the pedestrian is in.
                    boxes = pid_annots[pid]['bbox']
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion']
                    cb = pid_annots[pid]['behavior']['CrossingBehavior'] # CrossingBehavior
                    for i in range(len(cb)):
                        if cb[i] == -1: # CrossingIrrelevant
                            cb[i] = 5
                    ti = pid_annots[pid]['behavior']['TrafficInteraction'] # TrafficInteraction
                    pa = pid_annots[pid]['behavior']['PedestrianActivity'] # PedestrianActivity
                    ai = pid_annots[pid]['behavior']['AttentionIndicators'] # AttentionIndicators
                    sd = pid_annots[pid]['behavior']['SocialDynamics'] # SocialDynamics
                    sb = pid_annots[pid]['behavior']['StationaryBehavior'] # StationaryBehavior
                    assert len(cb) == len(ti) == len(pa) == len(ai) == len(sd) == len(sb)
                    beh_labels = [[cb_,ti_,pa_,ai_,sd_,sb_] for cb_,ti_,pa_,ai_,sd_,sb_ in zip(cb,ti,pa,ai,sd,sb)]
                    # Get only the frames for which the height of the bboxes is in the
                    # specified range.
                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions, beh_labels = self._beh_filter(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions,beh_labels,occluded=False)
                

                    image_seq.extend(images[::seq_stride])
                    box_seq.extend(boxes[::seq_stride])
                    beh_label_seq.extend(beh_labels[::seq_stride])

                    assert len(image_seq) == len(box_seq) == len(beh_label_seq)
                    

        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        output = {'image': image_seq,
                'bbox': box_seq,
                'beh_label': beh_label_seq}

        with open("iddp_beh.pkl","wb") as f:
            pickle.dump(output,f)
        # Every element in the dictionary is a list of lists. Each list corresponds to a pedestrian track of interest.
        return output
