#!/bin/bash
currdate=`date +%Y%m%d-%H`
source ~/.bashrc2
source activate tf-keras


runfile='simpleNet_training_rawdata_vxvz'
datapath='/media/wywdisk/VSPdata/data/haveinvx/layer2_haveinvx'
outname=${runfile}-${currdate}

python ${runfile}.py \
  -data_path ${datapath} \
  -out_name ${outname} \
  -nt 4000 \
  -nr 400 \
  -ns 51 \
  -nph 2 \
  -batch_size 2 \
  -epoch 4 \
  -ratio 0.5 
