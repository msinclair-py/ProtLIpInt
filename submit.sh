#!/bin/bash
#### CONFIG ####
JOBNAME=${1}
NCPUS=${2}
COMMONPATH=${3}
DCDPATH=${4}
FIRST=${5}
LAST=${6}
PSFPATH=${7}
SIM=${8}
OUTPATH="/Scr/msincla01/Protein_Lipid_DL"
################

qsub -pe linux_smp $NCPUS -N ${JOBNAME}_${FIRST}_${LAST} -j y -o /Scr/msincla01/Jobs/ << EOF
ray start --head
/Scr/msincla01/anaconda3/bin/python3 /Scr/msincla01/github/ProtLIpInt/training_data.py $COMMONPATH $PSFPATH -o $OUTPATH -d $DCDPATH -t $NCPUS -mp ray -ff $FIRST -lf $LAST -sim $SIM
EOF
