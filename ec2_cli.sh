#commands to experiment local processing on an EC2 instance
#build the docker image
docker build --no-cache -t cdl-data .

#first time data prep in the attached shell
docker run -it -v ~/hdd/data-all:/data/ -v "$(pwd)":/cdl_training_data/ -p 8888:8888 cdl-data


cd /cdl_training_data/data/required_sources
wget https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2022_30m_cdls.zip
mkdir 2022_30m_cdls
unzip -j 2022_30m_cdls.zip -d 2022_30m_cdls
rm -f 2022_30m_cdls.zip

# background running
docker run --name cdl_data -v ~/hdd/data-all:/data/ -v "$(pwd)":/cdl_training_data/ -p 8888:8888 cdl-data
docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/create_bb.py"
docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/prepare_async.py"

docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/create_CID_payload.py"

#offline only: download raw data for IPFS
docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/local_HLS_cache.py"

#reproject HLS to CDL
docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/reproject.py"

#crop chips as training set
docker exec cdl_data /bin/sh -c "cd /cdl_training_data && python crop_classification/data_prep/process_chips.py"

##setting up ipfs node

export ipfs_staging=/home/ec2-user/hdd/ipfs_all/staging
export ipfs_data=/home/ec2-user/hdd/ipfs_all/data
docker run -d --name ipfs_host -v $ipfs_staging:/export -v $ipfs_data:/data/ipfs -p 4001:4001 -p 4001:4001/udp -p 127.0.0.1:8080:8080 -p 127.0.0.1:5001:5001 ipfs/kubo:latest

## data pinning

do
    # Run the ipfs pin add command with the file_name and ipfs_source
    # echo "ipfs pin add -r --progress -n \"$file_name\" \"$ipfs_source\""
    timeout 30s ipfs pin add -r --progress -n "$file_name" "$ipfs_source"
    if [ $? -eq 124 ]
    then
        echo "Error: ipfs pin add command timed out for "$file_name" | "$ipfs_source""
    fi
done

#!/bin/bash

# Replace with the path to your JSON file
#json_file="files.json"
json_file="data/misc/selected_cid_mapping.json"

jq -r 'to_entries[] | "\(.key) \(.value)"' "$json_file" | while read -r filename CID; do
    echo "Processing: $filename $CID"
    ipfs pin add -r --progress -n "$filename" "$CID"
done

ipfs add -r --cid-version=1 data/required_sources/2022_30m_cdls
#bafybeifv4drnsveeen524wwcbngsodg5ll737wv7uptourmaxgikkjsgqa

#dockerhub image push
#create repo at dockerhub then push local images

docker tag cdl-data zliu1208/crop-classification-training-data

bacalhau docker run \
--entrypoint ls \
-i src=ipfs://bafybeifv4drnsveeen524wwcbngsodg5ll737wv7uptourmaxgikkjsgqa,dst=/data/required_sources/2022_30m_cdls \
--output "required_sources:/data/required_sources" \
--publisher ipfs \
zliu1208/crop-classification-training-data:latest \
/data/required_sources/2022_30m_cdls


