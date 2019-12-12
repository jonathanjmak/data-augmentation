# Data Augmentation Techniques and Style Shifting GANS
Utilize various data augmentation techniques and explore accuracy prediction on CIFAR-10.

# Instructions
First train a conditional gan:
```
cd cgan/
python main.py
```

Then generate samples from the trained conditional gan:
```
python sample.py
```

Finally train the classifier:
```
cd ../vggnet/
python main.py
```

You can additionally train a discriminator to differentiate real from generated images:
```
cd ../cgan/
python threshold.py
```

Then use it to generate synthetic datasets that have a "realness" above a certain threshold:
```
python sample.py --threshold=0.9
```

Then train the classifier with this thresholded dataset:
```
python main.py --p_cifar=1. --p_thresholded=1. --threshold=0.9
```
