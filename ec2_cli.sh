#commands to experiment local processing on an EC2 instance
#build the docker image
docker build --no-cache -t cdl-data .

#first time data prep in the attached shell
docker run -it -v ~/hdd/data-all:/data/ -v "$(pwd)":/cdl_training_data/ -p 8888:8888 cdl-data


cd /cdl_training_data/data/required_sources
wget https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2022_30m_cdls.zip
mkdir 2022_30m_cdls
unzip 2022_30m_cdls.zip -d 2022_30m_cdls

# background running
docker run --name cdl_data -v ~/hdd/data-all:/data/ -v "$(pwd)":/cdl_training_data/ -p 8888:8888 cdl-data
docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/create_bb.py"


#python crop_classification/data_prep/create_bb.py