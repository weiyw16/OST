#!/bin/bash
currdate=`date +%Y%m%d-%H`
source ~/.bashrc2
source activate tf-keras


runfile='GAN-training-3c-parse'
datapath='/home/wyw/data/SEAM_I_walkaway_vsp_s23900/SEAM_Well1VSP_Shots23900.sgy'
outname=${runfile}-${currdate}

python ${runfile}.py \
  -data_path ${datapath} \
  -out_name ${outname} \
  -nt 2001 \
  -nr 467 \
  -ns 151 \
  -nph 3 \
  -nop 1 \
  -batch_size 2 \
  -sample_interval 10 \
  -epoch 4 \
  -ratio 0.5
