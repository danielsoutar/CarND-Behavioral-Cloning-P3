#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./centre_jungle.jpg "Centre of track, centre angle"
[image2]: ./left_image.jpg "Left image for recovery"
[image3]: ./right_image.jpg "Right image for recovery"
[image4]: ./centre_image.jpg "Centre image for recovery"
[image5]: ./examples/ "Normal Image"
[image6]: ./examples/ "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* nvidia-crop-aug-flags-final.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* climb_dance.mp4 containing the output video verifying that my model can complete a lap of the track without veering off

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with the same architecture as Nvidia's (clone.py lines 73-89). 

The model includes RELU layers to introduce nonlinearity (lines 78-82), and the data is initially normalized in the model using a Keras lambda layer (line 76). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 10-16), and for each layer I used L2-regularisation on the weights to reduce variance and speed up learning (code lines 90-94, 96-98). The amount of data that I used ended up being too much to fit into memory all at once, so instead I partitioned the data into several sets (including the harder track), and would load that as well as potentially a saved Keras model to train further. I am aware Python generators are better suited for such a task (since the amount of data I used was such that I had issues with the swap space - Python was using 10GB of memory at one point!). But Python generators were not working for me, nor were they as intuitive. At any rate, this approach worked perfectly fine and meant I could have greater control at the command line, which in my head made things a little clearer. One sneaky benefit of training the network like this was that I essentially instilled a bias into the network to drive straight around the centre by first training it on a dataset consisting solely of video from driving smoothly around the centre - additionally, I specifically recorded driving around the individual corners to improve the balance of the dataset. Then I trained the same network on a smaller dataset of recovery-based video, which effectively equated to fine-tuning it on this dataset. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, often with amusing results, it should be added ;)

####3. Model parameter tuning

The model used an adam optimizer with a lower learning rate than the default since I didn't want the network to overfit the centre dataset (others have said such a move seemed to help them as well), with a mean-squared error loss, so the learning rate was not tuned manually (clone.py line 99).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving, recovering from the left and right sides of the road, and driving in reverse. The data collection was definitely the most important step - even with more powerful architectures, I had experienced worse performance due to too much data and poor generalisation. So this was crucial to get right.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off with the basic ones found from the Udacity lectures and work upwards, with a basic one just to prototype and get the ball rolling. A convolutional neural network seemed like a natural fit for this task owing to their excellent performance on image-related tasks, so that's what I used. After that I would use the same data - partitioned into training and validation sets - on several variations of the model to see if there was any improvement. This was made somewhat less useful by the confusing error messages that I ended up getting due to excessively large amounts of data (ran out of memory on AWS because of this, which was fairly annoying). I resolved this by saving the data into separate folders (e.g. driving around the track clockwise was separated from driving counter-clockwise), training the model partially with a particular dataset using command-line arguments, and then reloading that model with other datasets. This worked quite well and prototyping was easy at this stage.

There were a couple of occasions where my car amazingly managed to drive straight along the first straight, onto the little stone jetty before tragically drowning. I resolved this chiefly by paying close attention to the dataset containing video of driving away from edges towards the centre (and to help with this I had included the side images to reinforce this behaviour).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

As above, the Nvidia architecture was deemed powerful enough (as one would hope), and I used this to great effect. The architecure consists of:

 * A lambda layer to normalise the data
 * I cropped the image size to fit into the network
 * I then had several convolutional layers, all using ReLu activation functions, with the first 3 layers performing a subsampling to reduce sensitivity to noise and with a 5x5 filter, the latter without subsampling (as the information was compressed enough at this stage), with 3x3 filters. The number of feature channels grows throughout as this captures higher levels of complexity in data.
 * I flatten the result, and then subject this one-dimensional output to several dense layers (i.e. gigantic matrix multiplications, albeit without any activations by default).
 * Finally I get a single output, which is the prediction for the steering angle.
 
 * How did you decide on the number and type of layers? Why is this model suitable for the problem at hand?
 The exact number of layers is very much a trial-and-error thing, but roughly the idea was to start off with several convolutional layers that naturally fit to image data. Convolutional layers essentially assume a relationship between neighbouring data, and since image data is very much relational (pixels tend to have a strong relationship with neighbouring ones in images), that is why they suit this kind of problem of detecting and recognising the road, and then trying to angle the car to keep it in the centre. Generally, one tends to have several convolutional layers to promote a hierarchy of features from the image - the first layer captures lines and edges, the second one basic shapes, the third one more complex shapes on top of that, and so on. In my case I used 5 layers, so this allows the network to extract a fairly high-level understanding of the data before piping it through a series of conventional layers with regularisation and ReLu activation functions. These conventional ('Dense') layers don't necessarily assume a neighbouring relationship between data, but since the convolutional part of the network has mostly eliminated that by extracting high-level features, this isn't an issue. The dense layers also act as a way of compressing the high-level feature vector from the convolutional part of the network into a single-valued output for the car to act on. I used 4 layers as per the Nvidia architecture, with the number of nodes decreasing. This allows the feature vector to be smoothly compressed into a single prediction with minimal information loss.
 
 * How did you evaluate the final model?
 I didn't end up testing it on a test set primarily because I wanted to put this number in context - the validation loss and training loss were generally accurate enough to indicate whether the loss was appropriately low, and instead I tested it on the actual simulator. By doing this I was able to actually tell whether my model was robust - certain models were low on both training and validation errors but ended up driving off road. Arguably a test set would have been even more suggestive, but it wouldn't have been as useful since it wouldn't necessarily have told me where exactly the model went wrong with regards to the track. Besides, I wanted to test it on the first track, and all my datasets involved a mixing of both tracks, so although getting the right result on the jungle track was useful my goal was best served by seeing the results of the model on the simulator live. The final model ended up being the one that successfully managed to go around the track with nary a flutter - and successfully resolving the slight off-lane tendencies of the first model I tried.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four or so laps on the first track using centre lane driving. Here is an example image of center lane driving on the bonus track:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to centre so that the vehicle would avoid going off road. These images show what a recovery looks like starting from the right side of the track:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two (albeit with only 2 laps as opposed to 4) in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help promote symmetry invariance, along with a small correction factor of 0.1. I didn't end up using grayscale images primarily because I felt the colour of the road would be a useful feature to contrast with the rest of the image. I also narrowed the image to a smaller region, since a car doesn't really need to see the sky or look above in order to drive. I used a random shuffling of the data set and put 20% of the data into a validation set.

