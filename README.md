# Flag_Detection

# Aim and Objective

# AIM

To create a Flag detection system which will detect the flag and  recommended the name of the country of that flag


# OBJECTIVE

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module that a particular flag belong to which country

# ABSTRACT

• A name of country of flag can be detected by the live feed derived from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML),
 where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• Flags are not designed to be used basically with automatic identification purposes. Normally flags have that attributes those are used for identification on a national basis.Flags which are regarded as the symbols of countries have various colors and symbols. Each color, symbol, coat of arms and emblem used in flags has a specific meaning.

• flags have their own meanings anddifferent country flags hold history and memory of significant people and events.every country experienced hardships in the past and most of them fought for theirfreedom and equality. So identifying national flag has enormous importance to every people

# Introduction

• This project is based on a Window detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about the flag to  identify which nation belongs the flag, .i.e., Italy ,Russia,Canada,Japan,India,France.

• Flags can be classified into different countries  based on the image annotation we give in roboflow.

• Specifically, the nationalflag detection requires lots of study to classify and differentiate the inherentuncertainty. In a Flags because of various shapes and  colors , Sometimes it is difficult to detect various shapes on it for ex: - stars,dots,leafs,circles. The detection of these characteristics  results in making the model better trained.

• Also, the shape on flags sometimes makes it difficult for the model to realize the  difference between the name of different  countries.

• Neural networks and machine learning have been used for these tasks and have obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for flag detection  as well.

# Literature Review

• Flags are not designed to be used basically with automatic identification purposes. Normally flags have that attributes those are used for identification on a national basis.

• This implies, there is typically no simplified algorithm by which theseattributes can be consolidated to identify which nation belongs the flag

• Flags of the world are more than just colorful displays. They show pride for their nations and have history and backgrounds to them. Independence is shown through the flag as a symbol; the colors of many flags relate to their freedom and beliefs. Common traits are found in many flags because of similar histories of the countries. 

• Historically, flags originated as military standards, used as field signs. 

• Flags that are regarded as the symbols of countries incorporate many colors, symbols, coats of arms and emblems. Each color, symbol, coat of arms and emblem used in flags has a specific meaning. Nations present their social and cultural values (such as Patriotism, Solidarity, Honesty, Altruism, Humanism, Optimism, Tolerance, Harmony, Dignity, Fervor, Loyalty, Heroism, Wisdom, Generosity, Nobility, Virtuousness, Purity)

# Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Proposed System

1.Study basics of machine learning and image recognition.

2.Start with implementation

➢ Front-end development ➢ Back-end development

3.Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify the clarity of windows.

4.Use datasets to interpret the windows and suggest whether the windows are clear or unclean.

# Methodology

The flag detection system is a program that focuses on implementing real time flag detection.

It is a prototype of a new product that comprises of the main module:

Flag detection and then showing on viewfinder the 

 Detection Module

This Module is divided into two parts:

1] Flag detection

• Ability to detect the Flag in any input image or frame. The output is the bounding box coordinates on the detected Flag and shows country name of that flag.

• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

• This Datasets identifies Flag in a Bitmap graphic object and returns the bounding box image with annotation of Flags present in each image.

2] Country Name identification

• Classification of the Flags based on Country names

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

• YOLOv5 was used to train and test our model for various classes like India,france,italy,russia,japan,canada etc. We trained it for 150 epochs and achieved an accuracy of approximately 92%.


# Setup

# Installation

Initial Setup

Remove unwanted Applications.

      sudo apt-get remove --purge libreoffice*
      sudo apt-get remove --purge thunderbird*


 Create Swap file
      sudo fallocate -l 10.0G /swapfile1
      sudo chmod 600 /swapfile1
      sudo mkswap /swapfile1
      sudo vim /etc/fstab
      
      #################add line###########
      /swapfile1 swap swap defaults 0 0

Cuda Configuration

vim ~/.bashrc
     
     
     #############add line #############
     export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
     export
     LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
     export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
     
     
     source ~/.bashrc
     
     
  Update & Upgrade

        sudo apt-get update
        sudo apt-get upgrade
     
  Install some required Packages

        sudo apt install curl
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        sudo python3 get-pip.py
        sudo apt-get install libopenblas-base libopenmpi-dev

        sudo pip3 install pillow
        
 Install Torch

         curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
         mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
         sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

         #Check Torch, output should be "True" 
         sudo python3 -c "import torch; print(torch.cuda.is_available())
         
   Installation of torchvision
   
          git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
          cd torchvision/
          sudo python3 setup.py install
     
  Clone Yolov5

          git clone https://github.com/ultralytics/yolov5.git
          cd yolov5/
          sudo pip3 install numpy==1.19.4

          #comment torch,PyYAML and torchvision in requirement.txt

          sudo pip3 install --ignore-installed PyYAML>=5.3.1
          sudo pip3 install -r requirements.txt
          
Download weights and Test Yolov5 Installation on USB webcam

          sudo python3 detect.py
          sudo python3 detect.py --weights yolov5s.pt  --source 0

     
   
 # Flag Dataset Training
 
#We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder link of project.


# Running Flag Detection Model

source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0


# Advantages

➢ There are Total 195 countries in the whole world and with 195 Countries there are 195 different types of flags of Different shapes,size,colors and symbols.
   It is very much difficult for a peoples to remember this all these flags that the flag belongs to whic country, so it is very Usefull model for detecting flag.

➢It is very usefull for  tourist and many peoples.

➢ Just place the viewfinder showing the Flag on screen and it will detect it and shows the name of the name of the country the flag belongs to.


# Conclusion

Some remarkable results have been obtained as a result of investigating the meanings and the reasons behind the use flags utilized as symbols of countries and designed with various symbols and colors. Investigating the meaning of country flags shows that majority of them include geographical elements. Nations have replaced in their flags the physical and human geography elements that are important for them. Most of the world countries (86%) include elements related to geographical characteristics. The dominant elements in the natural and human environmental conditions have been emphasized in flags. Flags help us to perceive the geographical characteristics of the countries. Disciplines other than geography may contribute to science by undertaking research about flags as well.

     


