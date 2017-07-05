# deepGroup
Hierarchical deep neural network for activity analysis of surveillance videos at individual, group and overall level.

1. This is a research quality code, so don't expect it to work straightaway. However, I'll be delighted if someone uses my work. So you are encouraged to ask for help if you run into trouble raising an issue or contacting me via e-mail.

2. Please cite our work if you use codes or ideas from this repository. A bibtex style citation can be found in [deepgroup-bibtex.bib](deepgroup-bibtex.bib)

3. Both MATLAB and Python are used. Install appropriate python packages whenever needed (all required packages can be obtained from pip, conda etc).

4. Install keras https://keras.io/

5. Throughout this repository, collective activity dataset is referred to http://www-personal.umich.edu/~wgchoi/eccv12/wongun_eccv12.html
Reference:
A Unified Framework for Multi-Target Tracking and Collective Activity Recognition
ECCV, 2012, accepted as an Oral Presentation
W. Choi and S. Savarese

# Overview
In total, we have 3 levels of hierarchy and 5 main stages of computation. We assume that individual tracklets are already known. The five stages are:

1. Orientation
2. Individual action
3. Group detection
4. Group activity
5. Scene activity

See individual folders for each stage.

