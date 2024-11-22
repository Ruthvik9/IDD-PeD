"""
Interface for the IDD-Pedestrian dataset:

Ruthvik.

"""
import cv2
import csv
import sys
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
        image_set_nums = {'train': ['gp_set_0001','gp_set_0002','gp_set_0004','gp_set_0006'],
                          'val': ['gp_set_0003','gp_set_0005'],
                          'test': ['gp_set_0003','gp_set_0005'],
                          'all': ['gp_set_0001', 'gp_set_0002','gp_set_0003','gp_set_0004','gp_set_0005','gp_set_0006']}
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




    # def extract_and_save_images(self, extract_frame_type='annotated',lib='opencv'):
    #     """
    #     Extracts images from clips and saves on hard drive
    #     :param extract_frame_type: Whether to extract 'all' frames or only the ones that are 'annotated'
    #                          NOOTE - To do: mention to disc spaces consumed.
    #     :param lib: The library to use for extracting images. Options are 'imageio' and 'opencv'.
    #     You can use 'opencv' for faster extraction if you have ffmpeg installed. Else, use the slower lib 'imageio'.
    #     """
    #     cams = [f for f in listdir(self._videos_path)]
    #     for cam in cams:
    #         print("############################################")
    #         print("Extracting and saving images for camera", cam)
    #         print("############################################")
    #         set_folders = [f for f in sorted(listdir(join(self._videos_path,cam)))]
    #         for set_id in set_folders:
    #             print('Extracting frames from', set_id)
    #             set_folder_path = join(self._videos_path,cam,set_id)
    #             if extract_frame_type == 'annotated':
    #                 extract_frames = self.get_annotated_frame_numbers(set_id)
    #             else:
    #                 extract_frames = self.get_frame_numbers(set_id) # NOOTE - to change. the get_frame_numbers func ie.

    #             set_images_path = join(self._iddp_path,"images",cam,set_id)
    #             for vid, frames in sorted(extract_frames.items()):
    #                 print(vid)
    #                 video_images_path = join(set_images_path, vid)
    #                 if exists(video_images_path):
    #                     print(f"Images for {set_id} and {vid} already exist. Skipping...")
    #                     continue
    #                 num_frames = frames[0]
    #                 frames_list = frames[1:]
    #                 if not isdir(video_images_path):
    #                     makedirs(video_images_path)
                    
    #                 if lib == 'imageio':
    #                     reader = imageio.get_reader(join(set_folder_path,vid+'.MP4'))
    #                     #vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
    #                     #success, image = vidcap.read()
    #                     frame_num = 0
    #                     img_count = 0
    #                     #if not success:
    #                     #    raise Exception('Failed to open the video {}'.format(vid))
    #                     for frame in reader:
    #                         if frame_num in frames_list:
    #                             self.update_progress(img_count / num_frames)
    #                             img_count += 1
    #                             if not isfile(join(video_images_path, "%05d.png") % frame_num):
    #                                 imageio.imwrite(join(video_images_path, "%05d.png") % frame_num, frame)
    #                         #success, image = vidcap.read()
    #                         frame_num += 1
    #                     if num_frames != img_count:
    #                         print(f'num images {num_frames} don\'t match image count {img_count}')
    #                     print('\n')
    #                 elif lib == 'opencv':
    #                     vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.MP4'))
    #                     success, image = vidcap.read()
    #                     frame_num = 0
    #                     img_count = 0
    #                     if not success:
    #                         raise Exception('Failed to open the video {}'.format(vid))
    #                     while success:
    #                         if frame_num in frames_list:
    #                             self.update_progress(img_count / num_frames)
    #                             img_count += 1
    #                             if not isfile(join(video_images_path, "%05d.png") % frame_num):
    #                                 cv2.imwrite(join(video_images_path, "%05d.png") % frame_num, image)
    #                         success, image = vidcap.read()
    #                         frame_num += 1
    #                     if num_frames != img_count:
    #                         print(f'num images {num_frames} don\'t match image count {img_count}')
    #                     print('\n')

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
        intent_seq = []
        resolution_seq = []
        obds_seq, accT_seq, accX_seq, accY_seq, accZ_seq = [], [], [], [], []

        # For testing trajectory prediction for certain scenarios, append '_{category_name}' to 'test' as the image_set.
        set_ids, _pids = self._get_data_ids(image_set, params)
        
        if len(set_ids) != 5: # Only equal when data_split_type is not default.
            print("Considering the sets",set_ids,"for",image_set) # data_split_type is 'default'
        else:
            category = image_set.split('_')[1]
            print("Considering pedestrians of category - {} for testing.".format(category))

        if '_' in image_set:
            # True if image_set is changed for the certain categories.
            # Now that we've gotten the ped_ids we want, we can make image_set one of 'train', 'val', 'test'.
            image_set = image_set.split('_')[0]

        # set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
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

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent_seq.append(intent[::seq_stride])

                    resolutions = [[img_width, img_height]] * len(boxes)
                    resolution_seq.append(resolutions[::seq_stride])

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
