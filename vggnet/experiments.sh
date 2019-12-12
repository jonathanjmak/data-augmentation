#!/bin/bash
set -e
###
python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.90 --p_monet=1.00 --p_udnie=1.00


python main.py --p_cifar=1.00 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=1.00

python main.py --p_cifar=1.00 --p_thresholded=0.00 --p_monet=0.00 --p_udnie=1.00
python main.py --p_cifar=1.00 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=0.00


python main.py --p_cifar=0.00 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=1.00
python main.py --p_cifar=0.00 --p_thresholded=0.00 --p_monet=0.00 --p_udnie=1.00
python main.py --p_cifar=0.00 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=0.00


python main.py --p_cifar=0.01 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=1.00

python main.py --p_cifar=0.01 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=0.00
python main.py --p_cifar=0.01 --p_thresholded=0.00 --p_monet=0.00 --p_udnie=1.00

python main.py --p_cifar=0.10 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=1.00


python main.py --p_cifar=0.10 --p_thresholded=0.00 --p_monet=0.00 --p_udnie=1.00
python main.py --p_cifar=0.10 --p_thresholded=0.00 --p_monet=1.00 --p_udnie=0.00
###

# python main.py --p_cifar=0.01 --p_thresholded=0.00
# python main.py --p_cifar=0.01 --p_thresholded=0.10 --threshold=0.90
# python main.py --p_cifar=0.10 --p_thresholded=0.00 
# python main.py --p_cifar=0.01 --p_thresholded=1.00 --threshold=0.90

# python main.py --p_cifar=0.00 --p_thresholded=1.00 --threshold=0.00

# python main.py --p_cifar=0.00 --p_thresholded=1.00 --threshold=0.90

# python main.py --p_cifar=0.10 --p_thresholded=0.10 --threshold=0.90 

# python main.py --p_cifar=0.10 --p_thresholded=1.00 --threshold=0.90 

# python main.py --p_cifar=0.01 --p_thresholded=0.00 --augment
# python main.py --p_cifar=0.10 --p_thresholded=0.00 --augment


# python main.py --p_cifar=0.01 --p_thresholded=0.00
# python main.py --p_cifar=0.01 --p_thresholded=0.10 --threshold=0.90
# python main.py --p_cifar=0.01 --p_thresholded=1.00 --threshold=0.90


# python main.py --p_cifar=0.00 --p_thresholded=1.00 --threshold=0.00 
# python main.py --p_cifar=0.00 --p_thresholded=1.00 --threshold=0.90 

# python main.py --p_cifar=0.10 --p_thresholded=0.00 
# python main.py --p_cifar=0.10 --p_thresholded=0.10 --threshold=0.90 
# python main.py --p_cifar=0.10 --p_thresholded=1.00 --threshold=0.90 

# python main.py --p_cifar=1.00 --p_thresholded=0.0 
# python main.py --p_cifar=1.00 --p_thresholded=0.10 --threshold=0.90 

# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.00 
# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.10 
# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.50
# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.90 

# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.00 --augment
# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.10 --augment
# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.50 --augment
# python main.py --p_cifar=1.00 --p_thresholded=1.00 --threshold=0.90 --augment