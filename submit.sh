#!/bin/bash
#### CONFIG ####
#JOBNAME=${1}
#COMMONPATH=${2}
#DCDPATH=${3}
#FIRST=${4}
#LAST=${5}
#PSFPATH=${6}
#SIM=${7}
OUTPATH="/Scr/msincla01/Protein_Lipid_DL"
################

#qsub -pe linux_smp 48 -N ${JOBNAME}_${FIRST}_${LAST} -j y -o /Scr/msincla01/Jobs/ << EOF
#ray start --head
IFS=$'\n' read -d '' -r -a lines < submitter.txt

for i in "${lines[@]}"
do
	line=($i)
	COMMONPATH=${line[2]}
	DCDPATH=${line[3]}
	FIRST=${line[4]}
	LAST=${line[5]}
	PSFPATH=${line[6]}
	SIM=${line[7]}
	
	/Scr/msincla01/anaconda3/bin/python3 /Scr/msincla01/github/ProtLIpInt/training_data.py $COMMONPATH $PSFPATH -o $OUTPATH -d $DCDPATH -ff $FIRST -lf $LAST -sim $SIM
done
#EOF
