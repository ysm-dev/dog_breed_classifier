
# Download data from URL
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar

# unzip
tar -xvf images.tar
tar -xvf annotation.tar

# make Records folder
mkdir Records
