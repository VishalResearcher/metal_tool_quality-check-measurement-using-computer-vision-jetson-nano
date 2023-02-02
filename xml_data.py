from textwrap import dedent
from lxml import etree
import glob
import os
import cv2
import time
import numpy

def CreateXMLfile(path,directory,file_name,bboxes,NUM_CLASS):
    boxes = []
    for bbox in bboxes:
        boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], NUM_CLASS])#, bbox[4], NUM_CLASS[int(bbox[5])]])
        #boxes.append([bbox, NUM_CLASS])#, bbox[4], NUM_CLASS[int(bbox[5])]])
    print(boxes)

    img = cv2.imread(path)
    os.chdir(directory)
    filename = 'savedImage.jpg'
    #cv2.imwrite(filename, img)  
    print("After saving image:")  
    print(os.listdir(directory))
  
    print('Successfully saved')


    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)


    img_name = "XML_"+file_name+".png"
    cv2.imwrite(img_name,img)


    annotation = etree.Element("annotation")

    folder = etree.Element("folder")
    folder.text = os.path.basename(os.getcwd())
    annotation.append(folder)




    filename_xml = etree.Element("filename")
    filename_str = img_name.split(".")[0]
    filename_xml.text = img_name
    annotation.append(filename_xml)


    path = etree.Element("path")
    path.text = os.path.join(os.getcwd(), filename_str + ".jpg")
    annotation.append(path)

    source = etree.Element("source")
    annotation.append(source)

    database = etree.Element("database")
    database.text = "Unknown"
    source.append(database)

    size = etree.Element("size")
    annotation.append(size)

    width = etree.Element("width")
    height = etree.Element("height")
    depth = etree.Element("depth")

    img = cv2.imread(filename_xml.text)

    width.text = str(img.shape[1])
    height.text = str(img.shape[0])
    depth.text = str(img.shape[2])

    size.append(width)
    size.append(height)
    size.append(depth)

    segmented = etree.Element("segmented")
    segmented.text = "0"
    annotation.append(segmented)



    for Object in boxes:
        class_name = Object[4]
        xmin_l = str(int(float(Object[0])))
        ymin_l = str(int(float(Object[1])))
        xmax_l = str(int(float(Object[2])))
        ymax_l = str(int(float(Object[3])))


        obj = etree.Element("object")
        annotation.append(obj)

        name = etree.Element("name")
        name.text = class_name
        obj.append(name)

        pose = etree.Element("pose")
        pose.text = "Unspecified"
        obj.append(pose)

        truncated = etree.Element("truncated")
        truncated.text = "0"
        obj.append(truncated)

        difficult = etree.Element("difficult")
        difficult.text = "0"
        obj.append(difficult)

        bndbox = etree.Element("bndbox")
        obj.append(bndbox)

        xmin = etree.Element("xmin")
        xmin.text = xmin_l
        bndbox.append(xmin)

        ymin = etree.Element("ymin")
        ymin.text = ymin_l
        bndbox.append(ymin)

        xmax = etree.Element("xmax")
        xmax.text = xmax_l
        bndbox.append(xmax)

        ymax = etree.Element("ymax")
        ymax.text = ymax_l
        bndbox.append(ymax)


        print(class_name,xmin_l,ymin_l,xmax_l,ymax_l)


        # write xml to file
    s = etree.tostring(annotation, pretty_print=True)
    with open(filename_str + ".xml", 'wb') as f:
        f.write(s)
        f.close()

    os.chdir("..")


#path = "/home/vishal/yolo-tensorflow/railway_metal_custom_data/yolov4-custom-functions/13.jpg"
#directory = "/home/vishal/yolo-tensorflow/railway_metal_custom_data/yolov4-custom-functions/newfolder"
#file_name = "new2"
#NUM_CLASS = 2
#bboxes = [[586,157,145,55],[11,153,12,123],[12,6,25,36],[587,10,96,47]]
#bboxes = [[586,157,145,55]]
#bboxes = pandas.array(data=[1,2,3,4],dtype=numpy.int8)
#CreateXMLfile(path,directory,file_name,bboxes,NUM_CLASS)


