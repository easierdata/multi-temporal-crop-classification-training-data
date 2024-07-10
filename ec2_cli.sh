#commands to experiment local processing on an EC2 instance

docker build --no-cache -t cdl-data .

docker run -it -v ~/hdd/data-all:/data/ -v "$(pwd)":/cdl_training_data/ -p 8888:8888 cdl-data

#in the attached shell
cd /cdl_training_data/data/required_sources
wget https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2022_30m_cdls.zip
mkdir 2022_30m_cdls
unzip 2022_30m_cdls.zip -d 2022_30m_cdls

cd /cdl_training_data
python crop_classification/data_prep/create_bb.py