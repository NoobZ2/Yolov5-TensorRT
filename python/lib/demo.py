import random

import cv2
import sys
import argparse

from matplotlib import colors

from Processor import Processor
from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-model',  help='trt engine file located in ./models', required=False)
    parser.add_argument('-image', help='image file path', required=False)
    args = parser.parse_args()
    model = args.model or 'yolov5s.trt'
    img = args.image or 'sample_720p.jpg'
    return { 'model': model, 'image': img }

def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    print('image arg', args['image'])
    #img = cv2.imread('inputs/{}'.format(args['image']))
    img = cv2.imread("/home/jiqing/jq/bottle/33/3 (3).jpg")
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
    #print(type(img))

    # inference
    #output = processor.detect(img)
    #img = cv2.resize(img, (640, 640))

        output = processor.detect(frame)
        img = cv2.resize(frame, (640, 640))

    # object visualization
        object_grids = processor.extract_object_grids(output)
    #visualizer.draw_object_grid(img, object_grids, 0.1)

    # class visualization
        class_grids = processor.extract_class_grids(output)
    #visualizer.draw_class_grid(img, class_grids, 0.01)

    # bounding box visualization
        boxes = processor.extract_boxes(output)
    #visualizer.draw_boxes(img, boxes)

    # final results
        boxes, confs, classes = processor.post_process(output)
    #print(classes)


    #label = f'{names[int(classes)]} {confs:.2f}'
        visualizer.draw_results(img, boxes, confs, classes)
    #plot_one_box(boxes, img, label=label, color=colors[int(classes)], line_thickness=3)
        #print("***************")

if __name__ == '__main__':
    main()   
