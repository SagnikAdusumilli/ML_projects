# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# function to locate and label images
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    #create a torch variable for the neural network
    frame_t = transform(frame)[0]
    # permute changes the order of color to grb (nn was trained this way)
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    # converting input into a batch because nn accepts input in batches
    x = x.unsqueeze(0)
    # no backward propogation as model is pre-trained 
    with torch.no_grad():
        y = net(x) #output of the nn
    detections = y.data
    #scale to normalize the postion of the detected objects
    scale = torch.Tensor([width, height, width, height])
    # detection = [batch, number of classes, number of occurances, (score, x0, y0, x1, y1)]
    for c in range(detections.size(1)):
        j = 0
        while detections[0, c, j, 0] > 0.6: #we only want high confidence
            pt = (detections[0, c, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)
            cv2.putText(frame, labelmap[c - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

#build load up the SSD nn
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location= lambda storage, loc: storage))

# create the transformation object
# the color values must be put into the color scale at which the nn was trained 
transfrom = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#object detection on the video 
reader = imageio.get_reader('./epic_horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output_60.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net, transfrom)
    writer.append_data(frame)
    print("processed frame: %d" % i)
writer.close()
