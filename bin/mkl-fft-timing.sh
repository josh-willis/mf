#!/bin/bash

NODE=`hostname -s`
export NODE

if [ -z ${VIRTUAL_ENV+x} ]; then 
    echo "Error: VIRTUAL_ENV is unset.  You must run inside a virtual environment pointing to PyCBC, multibench, and mf"; 
    exit 1;
fi

if [ -z ${OUTPUT_DIR} ]; then 
    echo "Error: OUTPUT_DIR is unset or empty.  You must set this variable to the prefix for the output files"; 
    exit 1;
fi

if [ -z ${INPUT_FILE} ]; then 
    echo "Error: INPUT_FILE is unset or empty.  You must set this variable to the location of the input file"; 
    exit 1;
fi

if [ -z ${PRECISION} ]; then
    echo "Error: PRECISION is not set; it must be either 'float' or 'double'";
    exit 1;
fi

if [ -z ${CPULIST+x} ]; then
    echo "Error: CPULIST is unset. You must set this variable to be a comma-delimited list of all cores to run together,"
    echo "with spaces separating the lists for different groups"
    echo "Example: export CPULIST='0,1,2,3 4,5,6,7' to have four threads together on cores 0-3, and four more on 4-7"
    exit 1;
fi

export NCPUS=`echo $CPULIST | cut -f1 -d" " | tr -d '\n' | tr ',' ' ' | wc -w`

if ! $(python -c "import pycbc.fft.mkl" 1>/dev/null 2>&1); then
    echo "Unable to import pycbc.fft.mkl; exiting";
    exit 1;
fi

export logfile=${OUTPUT_DIR}/${PRECISION}-${NCPUS}cpus-${NODE}-mkl.log

echo "VIRTUAL_ENV = $VIRTUAL_ENV" > $logfile
echo "NODE is ${NODE}" >> $logfile
echo "CPULIST = ${CPULIST}" >> $logfile
echo "NCPUS = $NCPUS" >> $logfile
echo "OUTPUT_DIR = $OUTPUT_DIR" >> $logfile
echo "INPUT_FILE = $INPUT_FILE" >> $logfile
echo "PRECISION = ${PRECISION}" >> $logfile

echo "" >> $logfile
echo "Location of pycbc.fft.mkl module:" >> $logfile
python -c "import pycbc.fft.mkl; print pycbc.fft.mkl.__file__" 1>>$logfile 2>/dev/null

echo "" >> $logfile
echo "Location of MKL shared library:" >> $logfile
python -c "import pycbc.fft.mkl; print pycbc.fft.mkl.lib" 1>>$logfile 2>/dev/null

echo "" >> $logfile
echo "Output of numactl --hardware:" >> $logfile
numactl --hardware  >> $logfile

echo "" >> $logfile
echo "lscpu:" >> $logfile
lscpu >> $logfile

echo "" >> $logfile
echo "CPU flags from /proc/cpuinfo:" >> $logfile
cat /proc/cpuinfo | grep flags | uniq >> $logfile

#exit 0

echo "" >> $logfile
echo "*****************************************************" >> $logfile
echo " MKL Benchmarks " >> $logfile
echo "*****************************************************" >> $logfile

many_problem_bench --mbench-nthreads-env-name PYCBC_NUM_THREADS --mbench-wait-time 30 --mbench-cpu-affinity-list \
${CPULIST} --clear-cache \
--mbench-timing-program pycbc_timing --mbench-dummy-program pycbc_dummy \
--mbench-input-file ${INPUT_FILE} \
--mbench-output-file ${OUTPUT_DIR}/${PRECISION}-${NCPUS}cpus-${NODE}-mkl.dat \
--mbench-input-arguments problem method --mbench-time 30.0 --mbench-repeats 15 \
--fft-backends mkl --processing-scheme mkl:env >> $logfile

