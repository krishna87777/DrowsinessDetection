#import torch
#from matplotlib import pyplot as plt
#import numpy as np
#import cv2


#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#cap = cv2.VideoCapture(0)
# while cap.isOpened():
#   ret, frame = cap.read(0)

# Make detections
# results = model(frame)

# cv2.imshow('YOLO', np.squeeze(results.render()))

# if cv2.waitKey(10) & 0xFF == ord('q'):
# break
# cap.release()
# cv2.destroyAllWindows()
#import uuid  # name the image
#import os
#import time

#IMAGES_PATH = os.path.join('data1', 'images')  # /data/images
##labels = ['awake', 'drowsy']
#number_imgs = 20
#for label in labels:
 #   print('Collecting images for {}'.format(label))
 #  time.sleep(5)

    # Loop through image range
  #  for img_num in range(number_imgs):
      #  print('Collecting images for {}, image number {}'.format(label, img_num))

        # Webcam feed
        #ret, frame = cap.read()

        # Naming out image path
        #imgname = os.path.join(IMAGES_PATH, label + '.' + str(uuid.uuid1()) + '.jpg')

        # Writes out image to file
        #cv2.imwrite(imgname, frame)

        # Render to the screen
        #cv2.imshow('Image Collection', frame)

        # 4 second delay between captures
        #time.sleep(2)

        #if cv2.waitKey(10) & 0xFF == ord('q'):
         #   break
#cap.release()
#cv2.destroyAllWindows()

#for label in labels:
 #   print('collecting images for {}', format(labels))
    # Loop through image range
  #  for img_num in range(number_imgs):
   #     print('Collecting images for {}, image number {}'.format(label, img_num))

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)
#cap = cv2.VideoCapture(0)
#while cap.isOpened():
 #   ret, frame = cap.read()

    # Make detections
  #  results = model(frame)
   # cv2.imshow('YOLO', np.squeeze(results.render()))

    #if cv2.waitKey(10) & 0xFF == ord('q'):
     #   break
#cap.release()
#cv2.destroyAllWindows()

import numpy as np
import tkinter as tk
import customtkinter as ctk
import torch
import cv2
from PIL import Image, ImageTk
import vlc
app = tk.Tk()
app.geometry("600x600")
app.title("Drowsy Boi 4.0")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=480,width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

counter = 0
counterLabel = ctk.CTkLabel(text=counter,height=40,width=120,font=("Arial",20),text_color="white",fg_color="teal",master=None)
counterLabel.pack(pady=10)
def reset_counter():
    global counter
    counter = 0
resetButton = ctk.CTkButton(text="Reset Counter",command=reset_counter,height=40,width=120,font=("Arial",20),text_color="white",fg_color="teal",master=None)
resetButton.pack()


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)
cap = cv2.VideoCapture(0)
def detect():
    global counter
    ret,frame = cap.read()
    frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())
    if len(results.xywh[0])>0:
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]

        if dconf.item() > 0.85 and dclass.item() == 16.0:
            p = vlc.MediaPlayer(f"C:/Users/91700/PycharmProjects/pythonProject8/call-to-attention-123107.mp3")
            p.play()
            counter+=1


    imgarr = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk=imgtk
    vid.configure(image=imgtk)
    vid.after(10,detect)
    counterLabel.configure(text=counter)

detect()
app.mainloop()
