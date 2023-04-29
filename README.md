# Tracking the ball in beach volleyball

## How to compile and execute project

1. Get a video of volleyball play with static recording(I.E. GoPro mounted on tripod)
2. Get images to train model by running getBall.py
3. Manually classify images into /volleyball-tracking/training/basedir/sort/ as ball and notball
4. Split 80% of ball and notball into train and 20% into validation
5. Run train.py 
6. Run model_loader.py
7. Run ballornotball.py


## Used 
- OpenCV
- Keras with Tensorflow

