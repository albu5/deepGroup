# Individual Actvity
Contains code to train and test individual activity recognition network.

==============================================
How to use:
==============================================
1. Run write_resnet_to_tracks.py to write features of images of individuals to a directory.
2. Run indiviudal_activity_data.m to read resnet features of individuals as well as their position (x,y,w,h) and save them for training and testing individual activity network.
3. Run person_activity_lstm.py to train and test the individual activity network.

==============================================
Alternate:
==============================================
1. Run getTrajFeat.m to extract slopes of trajectories and the corresponding action labels.
2. Train any classifier on generated trainX and trainY arrays.
