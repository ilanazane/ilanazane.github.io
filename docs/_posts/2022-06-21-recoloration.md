---
layout: post
title:  "Recoloring Black and White Images with a CNN"
date: 2022-06-21 13:15:36 -0400
---

# Intro

A few years ago, my uncle decided to take old photos of my grandparents that were stored in a box and scan them so we could have digital copies for safekeeping. Naturally, I thought it would be cool if we could also have colored versions of them as well, but that's a complicated process. I can imagine that in order to do that you need to know how to use photoshop well (my photoshop skills are a bit limited) and actually understand some color theory and a bit of history in order to make the images realistic and as accurate as possible. So, as a computer scientist my next thought was to use a neural network 
to do the job for me and better yet use it as a final project for my Intro to AI course. I actually did a second iteration of this project a few years later. 

# First Version 

The first version of the project required taking a colored photo and splitting it in half so that one side was still colored and the other side was black and white. Then, my group and I used k-means clustering in order to pick out the *n* most prominent colors and using those to recolor the black and white half. Here are the results: 

*insert k-means photo here* 

# Second Version 

Since the results weren't *amazing* from the first part,  we then moved onto using a basic neural network. We had kind of figured that the k-means wouldn't be the best algorithm to use, but it was a good starting point. The neural network we used had only three layers and we calculated all of the weight derivatives by hand. This was my very frst time using a neural network and coding one from scratch-- it was a tedious process, but it was needed to finally clear up any misunderstandings or confusion I had around neural networks. Here are the results from that neural network: 

*insert NN photo here* 

## Side note: up until that point I had watched so many youtube videos on neural networks (an especially good one from 3Blue1Brown) but there was always some confusion as to how they *actually* worked. The only way to clear up that fogginess was to all of the calculus by hand. It was hard to appreciate the learning experience in the moment, but it was totally necessary. 

# Thrid Version 

Okay, so fast forward like two years later and I have to do a final project for my Pattern Recognition and Classification (AAI 646) course in grad school. My professors only instructions were to create a project using any of the machine learning algorithms we had learned over the course of the semester... so pretty broad. I decided that this would be the perfect opportunity to revist my black and white image recoloring project, because although I had completed it, there was *a* *lot* of room for improvement. 

I worked on this project by myself for almost a month and created a new and improved version that used a Convolutional Neural Network (CNN) to recolor the images. 

*Insert photos* 
## Model

The CNN consisted of 12 convolutional layers using Tensorflow and Keras that took images of size 256 pixels x 256 pixles as input. In order to extract features I had the model use a kernel size of 3x3 and a stride of 2-- i.e the images are analyzed by a 3x3 window to see only nine pixels of the image at a time and the stride parameter dictated that the window moved by 2 pixels each time. At the end of this process though I was left with an image that was about half the height and width of the original image. To counteract this I had an upsampling layer at the end of the model in order to increase the dimensions of the image to what it originall was to prevent loss of information. 

