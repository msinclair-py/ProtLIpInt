#### CONFIG ####
JOBNAME=${1}
NCPUS=${2}
INP=${3}
DCD=${4}
FIRST=${5}
LAST=${6}
################

qsub -pe linux_smp $NCPUS -N ${JOBNAME}_${FIRST}_${LAST} -j y -o /Scr/msincla01/Jobs/ << EOF
cd "$PWD"
python /Scr/msincla01/github/ProtLIpInt/training_data.py $INP $JOBNAME -o $PWD -d $DCD -t $NCPUS -mp ray -ff $FIRST -lf $LAST
EOF
