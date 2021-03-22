"""

███████╗███████╗████████╗██╗   ██╗██████╗     ████████╗██╗  ██╗███████╗    ███████╗███╗   ██╗██╗   ██╗██╗██████╗  ██████╗ ███╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗
██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗    ╚══██╔══╝██║  ██║██╔════╝    ██╔════╝████╗  ██║██║   ██║██║██╔══██╗██╔═══██╗████╗  ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
███████╗█████╗     ██║   ██║   ██║██████╔╝       ██║   ███████║█████╗      █████╗  ██╔██╗ ██║██║   ██║██║██████╔╝██║   ██║██╔██╗ ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║
╚════██║██╔══╝     ██║   ██║   ██║██╔═══╝        ██║   ██╔══██║██╔══╝      ██╔══╝  ██║╚██╗██║╚██╗ ██╔╝██║██╔══██╗██║   ██║██║╚██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║
███████║███████╗   ██║   ╚██████╔╝██║            ██║   ██║  ██║███████╗    ███████╗██║ ╚████║ ╚████╔╝ ██║██║  ██║╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║
╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝            ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚══════╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝

"""

## With this dependencies, would be feasible to run the program in Google Collab

# CUDA DRIVERS in Ubuntu 20.04 (WSL2) - https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0

# !pip install pyyaml==5.1
# !pip install torch==1.7.1 torchvision==0.8.2
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# !pip install opencv


# Personal NOTES:
# Use object pre-trained object detection
# models in inference on KITTI-MOTS

##
# How to register COCO Fromat Datasets
# https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-coco-format-dataset


"""
██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗███████╗
██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝
██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ███████╗
██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ╚════██║
██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ███████║
╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import glob
import os
from fnmatch import fnmatch
import time
import re
import pickle
import warnings
from Reader import *


"""
██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝
                                                        
"""

"""
When converting: outputs["instances"].to("cpu") to string, the expected result is:

{'instances': Instances(num_instances=1, image_height=256, image_width=256, fields=[pred_boxes: Boxes(tensor([[167.9608, 182.4473, 184.4191, 195.6117]], device='cuda:0')), scores: tensor([0.7329], device='cuda:0'), pred_classes: tensor([8], device='cuda:0')])}
This method was used on first releases to create a file with the following format:
                                 |
   each line of file == id_frame |  num_objects ;  list_of_probabibilities ; list_of_classes
      that means. 1000 frames =  |
      file of 1000 lines         |
"""
# POST: When receiving the output, it makes a string conversions to obtain a simplified string and then use it in CSV
#       Input is a string retreived from the command:     outputs["instances"].to("cpu")
def output_to_csv_line(line):
    warnings.warn("Method (output_to_csv_line) has been deprecated in W3. Build another one!")
    # Get full string
    text = str(line)
    # delete endlines
    # TODO : This can be improved with a expression like: re.sub(r"[\n\t\s]*", "", text_type) or ''.join(text_type)
    text = text.replace('\n','').replace('\r','')
    # delete first part of the string, till num of instances
    nInstances1 = text[text.find("=")+1:]
    # should be now: 1, image_height=256, image_width=256, fields=[pred_boxes: Boxes(tensor([[111.0964, 140.7526, 130.6315, 158.4013]])), scores: tensor([0.8170]), pred_classes: tensor([2])])
    nInstances2= nInstances1[:nInstances1.find(',')]
    # now we focus on looking other parts and repeat the logic till the end
    # probability:
    prob1 = nInstances1[nInstances1.find(' tensor(')+8:]
    prob2 = prob1[:prob1.find(']')+1]
    # classes:
    classes1 = prob1[prob1.find('pred_classes: tensor(')+21:]
    classes2 = classes1[:classes1.find(']')+1]
    """ #Debug
    print("Text = ", text)
    print("Ninstances = ", nInstances2)
    print("Precission = ", prob2)
    print("Classes = ", classes2)
    """
    final_result = nInstances2 + ";" + prob2 + ";" + classes2
    return final_result

# POST: In this 2nd version of the output to csv we use just scores, separated by whitespaces
#       This method is intended to be used in pd. format
def output_to_csv_line_only_scores(line):
    warnings.warn("Method (output_to_csv_only_scores) has been deprecated in W3. Build another one!")
    # Get full string
    text = str(line)
    # delete endlines
    # TODO : The following line can be improved with a expression like: re.sub(r"[\n\t\s]*", "", text_type) or ''.join(text_type)
    text = text.replace('\n','').replace('\r','')
    # delete first part of the string, till num of instances
    nInstances1 = text[text.find("=")+1:]
    # should be now: 1, image_height=256, image_width=256, fields=[pred_boxes: Boxes(tensor([[111.0964, 140.7526, 130.6315, 158.4013]])), scores: tensor([0.8170]), pred_classes: tensor([2])])
    nInstances2= nInstances1[:nInstances1.find(',')]
    # now we focus on looking other parts and repeat the logic till the end
    # probability:
    prob1 = nInstances1[nInstances1.find(' tensor(')+9:]
    prob2 = prob1[:prob1.find(']')]
    # classes:
    classes1 = prob1[prob1.find('pred_classes: tensor(')+21:]
    classes2 = classes1[:classes1.find(']')+1]

    '''
    print("Text = ", text)
    print("Ninstances = ", nInstances2)
    print("Precission = ", prob2)
    print("Classes = ", classes2)

    prob2 = prob2.replace(" ","")
    prob2 = prob2.replace(","," ")
    print("RESULT: ", prob2)
    '''
    final_result = prob2
    return final_result

# POST: Sugar for writting lists to files
def write_to_file(data, path):
    with open(path, 'w') as f:
        for i in data:
            print(i, file=f)

# POST: More sugar to write a var to .pckl format
# NOTE: I usually save pckl in the same directory, but if you prefer path should work too
#       End your files / path with: .pckl
def save_as_pickle(name_path, var):
    f = open(name_path, 'wb')
    pickle.dump(var, f)
    f.close()

# POST: Given a PATH and a file pattern (e.g: *jpg) it creates you a list of image_paths
def list_images_from_path(path, pattern):
    im_paths = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                im_paths.append(os.path.join(path, name))
    print( len(im_paths), " have been found in " + path + " with a given extension: "+ pattern)
    return im_paths


# POST: Given tot ticks, it shows some statits
def calculate_performance(t1, t2, nImages):
    print ("TIME ELAPSED:\t", t2-t1)
    print ("AVG time for img:\t", (t2-t1)/nImages)


# POST: Encapsulating the generation
#       Which means this is the Detectron class constructor
# TODO: Check which signature and parameters are required to filter the output.
# TODO: I don't know at the time of writting these lines if is it possible to filter by size, or class. Maybe yes.
def generate_predictor(threshold, model):
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file(model))
    # AIXO ES IMPORTANT! Si baixes el threshold, et surtiran més deteccions
    # TODO: Review the following line. It has been used untill now changing the model.... but I don't know
    # TODO: what might happen when modifying this internal .RETINANET & .ROI_HEADS attributes
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)

    # print ("A model of type: " + str(model) + " with Threshold: " + threshold + " will be used")

    return cfg, predictor

# POST: It manages the input of a loaded annotations file from .txt. Input is a given dictionary
#       The output format of the annotation is: id_frame; bbox; category_id
#
# TODO: NEW! Needs adaptation?
# TODO: As it's an annotation... the score of each bbox should be 1 = 100%
def manage_annotations(input):
    output = []
    for dict in input:
        frame_related = dict["image_id"]
        for annot in dict["annotations"]:  #
            type = annot["category_id"]
            bbox = annot["bbox"]

            output.append([frame_related,bbox,type])

    #print ("Final annotations have been processed")
    #print ("With a size of: " + str(len(load)))

    return output

# PRE:  input should be: [id_frame, prediction_from_detectron]
# POST: It manages the transformation of different Instances present in a list with its given id_frame
#       As Instances are generated one for each frame with the predictor
#       This method splits each object and bbox to a unique line
#       So the format will be:
#         id_frame, bbox, score, class  [similar to the annotation one]
def manage_predictions(input):
    output = []

    ## input should be: [id_frame, prediction_from_detectron]
    for index, line in input:
        type = line["instances"].get("pred_classes")
        bbox = line["instances"].get("pred_boxes")
        scores = line["instances"].get("scores")

        types = []
        bboxes = []
        score_list = []

        # Treat each type of data
        # type of class (aka. pred_classes)
        text_type = str(type)                       # Noisy text: tensor at beginning + cuda at the end
        t10 = re.sub(r"[\n\t\s]*", "", text_type)   # Removing all spaces. Equivalent to ''.join(text_type)
        t11 = t10.replace("tensor([", "")           # Deleting first part
        t12 = t11.replace("],device=\'cuda:0\')", "")   # Deleting last part
        regex_pattern = '\d{1,3}(?:,\d{3})*'  # REGEX: 1 to 3 digits; followed by 0 or more the non capture group comma followed by 3 digits - https://www.reddit.com/r/regex/comments/90g73v/identifying_comma_separated_numbers/
        for match in re.finditer(regex_pattern, t12):
            sGroup = match.group()
            types.append(sGroup)

        # bbox
        text_bbox = str(bbox)
        b10 = re.sub(r"[\n\t\s]*", "", text_bbox)   # Removing all spaces. Equivalent to ''.join(text_type)
        b11 = b10.replace("tensor([", "")           # Removing just the first "brackets" of tensor([
        regex_pattern = '\[(.*?)\]'                 # and taking content between brackets of the rest
        for match in re.finditer(regex_pattern, b11):
            sGroup = match.group()
            bboxes.append(sGroup)

        # scores
        text_scores = str(scores)                   # Very similar to first case, types.
        s10 = re.sub(r"[\n\t\s]*", "", text_scores)
        s11 = s10.replace("tensor([", "")
        s12 = s11.replace("],device=\'cuda:0\')", "")
        regex_pattern = '[+-]?([0-9]*[.])?[0-9]+'  # REGEX: https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
        for match in re.finditer(regex_pattern, s12):
            sGroup = match.group()
            score_list.append(sGroup)

        ## ASSERT All list should have the same length
        if (len(score_list)!=len(bboxes) or len(score_list)!=len(types)):       #score_list is the most buggy candidate. Check same size among others
            raise AssertionError("Some differences have been found in regex. Please, check the patterns")

        # generating the output with index == id_frame. Looping along internal lists of same length
        for index2 in range(len(types)):
            # frame_id = Retrieved by index of first loop; [bbox]; score; type
            row = [index, bboxes[index2], score_list[index2], types[index2]]
            output.append(row)

    return output

# POST: Given the annotations and predictions treated, it makes a comparisson.
# TODO: for now it's just a signature method.
def compare_both (annotations, predictions):

    write_to_file(annotations, "anot_out.txt")
    write_to_file(predictions, "pred_out.txt")


# IN TYPE 1: We use the first output format
def do_experiments_type1(cfg, predictor, train_images):
    results = []
    predictions = []
    for index, im_path in enumerate(train_images):
        im = cv2.imread(im_path)
        outputs = predictor(im)
        row = [index, outputs]
        # print (index)
        predictions.append(row)

    # we strip the predictions in order to have in a similar format to the annotations file
    predictions = manage_predictions(predictions)

    ## loading dataset annotations
    dataset_dicts = []
    # TODO: The following liens should be adapted
    # Load the dataset and read it
    anottations_dataset = "HEY, I NEED TO BE ADAPTED"       # here shoud be a call to receive the annotations from the AICITY CHALLENGE
    print(anottations_dataset)

    # then, convert it to same format as predictions
    annotations_dataset = manage_annotations(anottations_dataset)

    # TODO: Once you've created the annotations & loaded all.... you can save
    save_as_pickle("annotations.pckl", annotations_dataset)
    save_as_pickle("predictions.pckl", predictions)

    # TODO: If you've already save it and generate it before
    # TODO: Everything can be done more elegant by checking if a file of specific format is in path or not... and then
    #       loading or saving accordingly
    anottations_dataset = pickle.load(open('anotattions.pckl', 'rb'))
    predictions = pickle.load(open('predictions.pckl', 'rb'))

    # TODO: The method that will compare both frames_list
    compare_both(anottations_dataset, predictions)

# IN TYPE 2: We use the second output format, focused on weights with panda formats
def do_experiments_type2(cfg, predictor, train_images):
    results = []
    i = 0
    for im_path in train_images:
        im = cv2.imread(im_path)
        outputs = predictor(im)
        i+=1
        # Just if you need visualize it.....
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        results.append(output_to_csv_line_only_scores(outputs))

    return results

def write_results(path, results):
    # opening the csv file in 'w+' mode
    file = open(path, 'w+')
    for line in results:
        file.write(line + '\n')
    file.close()

"""
███████╗██╗  ██╗██████╗ ███████╗██████╗ ██╗███╗   ███╗███████╗███╗   ██╗████████╗███████╗
██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██╔══██╗██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔════╝
█████╗   ╚███╔╝ ██████╔╝█████╗  ██████╔╝██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████╗
██╔══╝   ██╔██╗ ██╔═══╝ ██╔══╝  ██╔══██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║
███████╗██╔╝ ██╗██║     ███████╗██║  ██║██║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ███████║
╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
                                                                                         
"""

# We like to compare two models. To do that, these are some Hypothesis we would like to check:

# Hypothesis 1: Which one of both identifies more correct elements? (considering same threshold, accuracy)
# Hypothesis 2: Which one of both identifies more type of elements? (wider recognition)
# Hypothesis 3: Is any more prone to errors? (false positive & false negatives)
# Hypothesis 4: Difference between using 1x ; 3x; Pyramids ; DC ?

# We are using MIT dataset, the same as in M3

PATH = "/home/mcv/datasets/KITTI-MOTS/"
PATTERN = "*.png"

# Get images
images = list_images_from_path(PATH, PATTERN)

# Models to analize
FASTER_RCNN = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
RETINANET = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
FASTER_RCNN_3 = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
RETINANET_3 = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

###
#  DIFFERENT EXPERIMENTS
#
## 1: FASTER_RCNN_3x @ 0.5
cfg, predictor = generate_predictor(0.5,FASTER_RCNN_3)
reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
gt = reader.get_annotations(classes=['car'])
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment1.csv',results)
calculate_performance(t0,t1, len(images))


print("Hey, I'm done here!")

