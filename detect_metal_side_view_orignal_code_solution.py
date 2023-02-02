import os
from tkinter import Radiobutton
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from object_detector_1 import *
import cv2
import math
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    # load model
    if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Fraame1',original_image)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        
        # get image name by using split method
        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if FLAGS.framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

##########################################################################################################
########################################CROP DETECTION PART ##############################################

        x1 = bboxes[0][0]
        y1 = bboxes[0][1]
        x2 = bboxes[0][2]
        y2 = bboxes[0][3]
        print("==============",x1, y1, x2, y2)
        cropped_image = original_image[int(y1):int(y2), int(x1):int(x2)]
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
##########################################################################################################



#########################################################################################################
################################ CRICLE FIND ###########################################################
        
  
        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))
  
        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 70,param2 = 34, minRadius = 13, maxRadius = 30)
        detector = HomogeneousBgDetector()
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 5))
        contours = detector.detect_objects(cropped_image)
        for c_0 in contours:
            rect = cv2.minAreaRect(c_0)
            (xa,ya),(w,h),angle = rect   #center point & width and hieght
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            print(box)
            x_box = sorted([box[0][0],box[1][0],box[2][0],box[3][0]])
            y_box = sorted([box[0][1],box[1][1],box[2][1],box[3][1]])
            print(x_box,y_box)

            epsilon = 0.01*cv2.arcLength(box, True)
            approx = cv2.approxPolyDP(box, epsilon, True)
            cv2.polylines(cropped_image, [box],True,(255,155,0),2)

###################################################################################################################

            #print(sort_val)


            cv2.line(cropped_image, (x_box[0],y_box[0]), (x_box[-1],y_box[1]), (255,0,0), 1)
            print("line1 :",x_box[0],y_box[0],x_box[-1],y_box[1])

            cv2.line(cropped_image, (x_box[0],y_box[0]), (x_box[-1],y_box[0]), (0,0,0), 1)
            print("line2 :",x_box[0],y_box[0],x_box[-1],y_box[0])







            x_axis_1 = x_box[0]-  x_box[-1]
            y_axis_1 = y_box[0] - y_box[1]
            

            dist1 = (x_axis_1*x_axis_1 + y_axis_1*y_axis_1)**0.5
            print("dist1   :",dist1)

            #straight line
            x_axis_2 = x_box[0] - x_box[-1]
            y_axis_2 = y_box[0]  - y_box[0]

            dist2 = (x_axis_2*x_axis_2 + y_axis_2*y_axis_2)**0.5
            print("dist2   :",dist2)


            m = dist2 / dist1

        #TAN0
            theta = (math.acos(m))
            degreesd1 =  math.degrees(theta) #theta
            print("--------------------:",theta)

          

            

            cos_val  = round(math.cos(theta),5)
            print("---------------------:",cos_val)






            #SIN0


            sin_vall  = round(math.sin(theta),3)
            print("-----------------------:",sin_vall)








##########################IF CRICLE IS PRESENT ########################################################    

  
            # Draw circles that are detected.
        if detected_circles is not None:
  
            #Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            new_sort_cricle = []
            new_sort_cricle_b = []
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                outputt_a = np.round(a).astype(int)
                # Draw the circumference of the circle.
                cv2.circle(cropped_image, (a, b), r, (0, 255, 0), 2)
                print(a,b,r)
                sort_cricle = int(outputt_a)
                new_sort_cricle.append(sort_cricle)


                outputt_b = np.round(b).astype(int)
                #sort_cricle_y = sorted(sort_cricle)
                new_sort_cricle_b.append(outputt_b)

            print(new_sort_cricle)
            print(new_sort_cricle_b)

            #-----------MOST IMPORTANT FORMULA-----------#
####################### FOR CREATE ,RUNING TIME ##############################

            for (point1,point2) in zip(new_sort_cricle,new_sort_cricle_b):
                print(point1,point2)
                D5 = ((point2-y_box[0]) - sin_vall*(point1-x_box[0]))/cos_val  
                #print(a,b,x_box[0],y_box[0])
                print("***********************",D5/2.68)

                output = D5/2.68
                AZ = int(point2-D5)
                #print(AZ)
######################################################################################

                cv2.line(cropped_image,(point1,AZ),(point1,point2), (0,1,255), 2)


                cv2.putText(cropped_image,"{}mm".format(round(output,1)),(point1,point2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,150))

          
            cv2.namedWindow("F", cv2.WINDOW_NORMAL)
            cv2.imshow('F',cropped_image)

        
        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = ['METAL']

        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

        # if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
        if FLAGS.ocr:
            ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)

        # if count flag is enabled, perform counting of objects
        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate = FLAGS.plate)
        else:
            image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate = FLAGS.plate)
        
        image = Image.fromarray(image.astype(np.uint8))
        #cv2.namedWindow("Fraame1", cv2.WINDOW_NORMAL)
        #cv2.imshow('Fraame1',image)
        if not FLAGS.dont_show:
            #image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(FLAGS.output + 'detection44' + str(count) + '.png', image)


            cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
            cv2.imshow('IMAGE',image)

            cv2.namedWindow("Fraame1", cv2.WINDOW_NORMAL)
            cv2.imshow('Fraame1',cropped_image)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
