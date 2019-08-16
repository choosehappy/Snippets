#----Approach using pytorch produced  docker image

docker pull pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
docker run -it  pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime bash


apt-get update


apt -y install libsm6 libxrender-dev libxtst6 nano libglib2.0-0 wget

pip install jupyter tables scikit-image scikit-learn numpy scipy matplotlib opencv-python tensorboardX



git clone https://github.com/choosehappy/PytorchDigitalPathology.git
wget http://andrewjanowczyk.com/wp-static/epi.tgz

tar -xzf epi.tgz
mkdir imgs
mv *.tif imgs/
cp PytorchDigitalPathology/segmentation_epistroma_unet/* .


jupyter nbconvert --ExecutePreprocessor.timeout=10000 --to notebook --execute make_hdf5.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=100000 --to notebook --execute train_unet.ipynb



#---Approach using empty ubuntu image
# docker run -it ubuntu bash  


# apt-get update


# apt-get -y install python3.6
# apt-get -y install git nano wget locate python3-distutils python3-pip
# apt-get -y install libsm6 libxrender-dev 
# updatedb &


# pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install jupyter tables scikit-image scikit-learn numpy scipy matplotlib opencv-python
# pip3 install tensorboardX


# cd /
# mkdir data
# cd data
# git clone https://github.com/choosehappy/PytorchDigitalPathology.git
# wget http://andrewjanowczyk.com/wp-static/epi.tgz

# tar -xzf epi.tgz
# mkdir imgs
# mv *.tif imgs/
# cp PytorchDigitalPathology/segmentation_epistroma_unet/* .


# jupyter nbconvert --ExecutePreprocessor.timeout=100000 --to notebook --execute make_hdf5.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=100000 --to notebook --execute train_unet.ipynb


