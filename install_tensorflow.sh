#!/bin/bash

wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh

bash Anaconda3-5.1.0-Linux-x86_64.sh -b -f

echo 'export PATH=/home/'$USER'/anaconda3/bin:$PATH' >> ~/.bashrc

~/anaconda3/bin/pip install 'tensorflow>1.2,<1.3'

~/anaconda3/bin/python -c 'import tensorflow as tf ; print(tf.__version__)'

echo 'Se na linha acima esta escrito 1.2.1, seu tensorflow esta funcionando corretamente'

~/anaconda3/bin/pip install opencv-python

rm Anaconda3-5.1.0-Linux-x86_64.sh