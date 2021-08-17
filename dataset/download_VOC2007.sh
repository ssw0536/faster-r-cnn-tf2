# make dataset directory
mkdir -p dataset/VOC2007/trainval
mkdir -p dataset/VOC2007/test

# get VOC2007 dataset
wget -O dataset/VOC2007/trainval/VOC2007-trainval.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar # VOC2007 train+val set
wget -O dataset/VOC2007/test/VOC2007-test.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar # VOC2007 test set

# unzip datasets
tar -xvf dataset/VOC2007/trainval/VOC2007-trainval.tar -C dataset/VOC2007/trainval
tar -xvf dataset/VOC2007/test/VOC2007-test.tar -C dataset/VOC2007/test

# delete tar files
rm -rf dataset/VOC2007/trainval/VOC2007-trainval.tar
rm -rf dataset/VOC2007/test/VOC2007-test.tar