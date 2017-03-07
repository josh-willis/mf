#!/bin/bash

NODE=`hostname -s`
export NODE

if [ -z ${VIRTUAL_ENV+x} ]; then 
    echo "Error: VIRTUAL_ENV is unset. You must run inside a virtual environment pointing to PyCBC, multibench, and mf"; 
    exit 1;
fi

if [ -z ${OUTPUT_DIR} ]; then 
    echo "Error: OUTPUT_DIR is unset or empty.  You must set this variable to the directory for the output files"; 
    exit 1;
fi

if [ -z ${INPUT_FILE} ]; then 
    echo "Error: INPUT_FILE is unset or empty.  You must set this variable to the location of the input file"; 
    exit 1;
fi

if [ -z ${FFTW_WISDOM_PREFIX} ]; then 
    echo "Error: FFTW_WISDOM_PREFIX is unset or empty.  You must set this variable"; 
    exit 1;
fi

if [ -z ${PRECISION} ]; then
    echo "Error: PRECISION is not set; it must be either 'float' or 'double'";
    exit 1;
fi

if [ -z ${FFTW_THREADS_BACKEND} ]; then
    echo "Error: FFTW_THREADS_BACKEND is not set; it must be either 'openmp' or 'pthreads'";
    exit 1;
fi

if [ -z ${MBWAIT} ]; then
    echo "Error: MBWAIT is not set; it must be the number of seconds to wait between starting dummy jobs";
    exit 1;
fi

if [ -z ${MBREPEAT} ]; then
    echo "Error: MBREPEAT is not set; it must be the number of times to repeat the test";
    exit 1;
fi

if [ -z ${MBTIME} ]; then
    echo "Error: MBTIME is not set; it must be the number of seconds to try to ensure each FFT loop lasts";
    exit 1;
fi

if [ -z ${CPULIST+x} ]; then
    echo "Error: CPULIST is unset. You must set this variable to be a comma-delimited list of all cores to run together,"
    echo "with spaces separating the lists for different groups"
    echo "Example: export CPULIST='0,1,2,3 4,5,6,7' to have four threads together on cores 0-3, and four more on 4-7"
    exit 1;
fi

export NCPUS=`echo $CPULIST | cut -f1 -d" " | tr -d '\n' | tr ',' ' ' | wc -w`

if ! $(python -c "import pycbc.fft.fftw" 1>/dev/null 2>&1); then
    echo "Unable to import pycbc.fft.fftw; exiting";
    exit 1;
fi

export logfile=${OUTPUT_DIR}/${PRECISION}-${NCPUS}cpus-${FFTW_THREADS_BACKEND}-${NODE}-fftw.log

echo "VIRTUAL_ENV = ${VIRTUAL_ENV}" > $logfile
echo "NODE = ${NODE}" >> $logfile
echo "CPULIST = ${CPULIST}" >> $logfile
echo "NCPUS = $NCPUS" >> $logfile
echo "OUTPUT_DIR = $OUTPUT_DIR" >> $logfile
echo "INPUT_FILE = $INPUT_FILE" >> $logfile
echo "PRECISION = ${PRECISION}" >> $logfile
export WISDOM_FILE="${OUTPUT_DIR}/${FFTW_WISDOM_PREFIX}-${PRECISION}-${NCPUS}cpus-${FFTW_THREADS_BACKEND}.wis"
echo "WISDOM_FILE = ${WISDOM_FILE}" >> $logfile
export OUTPUT_FILE="${OUTPUT_DIR}/${PRECISION}-${NCPUS}cpus-${FFTW_THREADS_BACKEND}-${NODE}"


echo "" >> $logfile
echo "Location of pycbc.fft.fftw module:" >> $logfile
python -c "import pycbc.fft.fftw; print pycbc.fft.fftw.__file__" 1>>$logfile 2>/dev/null

echo "" >> $logfile
echo "Location of FFTW single library:" >> $logfile
python -c "import pycbc.fft.fftw; print pycbc.fft.fftw.float_lib" 1>>$logfile 2>/dev/null

echo "" >> $logfile
echo "Location of FFTW double library:" >> $logfile
python -c "import pycbc.fft.fftw; print pycbc.fft.fftw.double_lib" 1>>$logfile 2>/dev/null

echo "" >> $logfile
echo "Output of numactl --hardware:" >> $logfile
numactl --hardware  >> $logfile

echo "" >> $logfile
echo "lscpu:" >> $logfile
lscpu >> $logfile

echo "" >> $logfile
echo "CPU flags from /proc/cpuinfo:" >> $logfile
cat /proc/cpuinfo | grep flags | uniq >> $logfile

echo ""  >> $logfile
echo "*****************************************************" >> $logfile
echo " FFTW Benchmarks " >> $logfile
echo "*****************************************************" >> $logfile
echo "                  Measure Level One        " >> $logfile
echo "****************************************************" >> $logfile

many_problem_bench --mbench-nthreads-env-name PYCBC_NUM_THREADS --mbench-wait-time ${MBWAIT} --mbench-cpu-affinity-list \
${CPULIST} --clear-cache \
--mbench-timing-program pycbc_timing --mbench-dummy-program pycbc_dummy \
--mbench-input-file ${INPUT_FILE} --mbench-output-file ${OUTPUT_FILE}-mlvl1.dat \
--mbench-input-arguments problem method --mbench-time ${MBTIME} --mbench-repeats ${MBREPEAT} \
--fftw-measure-level 1 --fftw-output-${PRECISION}-wisdom-file ${WISDOM_FILE} \
--fftw-threads-backend ${FFTW_THREADS_BACKEND} \
--fft-backends fftw --processing-scheme cpu:env >> $logfile
echo "'FFTW_MEASURE' planning" >> $logfile

echo "*****************************************************" >> $logfile
echo "                  Measure Level Two        " >> $logfile
echo "****************************************************" >> $logfile

many_problem_bench --mbench-nthreads-env-name PYCBC_NUM_THREADS --mbench-wait-time ${MBWAIT} --mbench-cpu-affinity-list \
${CPULIST} --clear-cache \
--mbench-timing-program pycbc_timing --mbench-dummy-program pycbc_dummy \
--mbench-input-file ${INPUT_FILE} --mbench-output-file ${OUTPUT_FILE}-mlvl2.dat \
--mbench-input-arguments problem method --mbench-time ${MBTIME} --mbench-repeats ${MBREPEAT} \
--fftw-measure-level 2 --fftw-input-${PRECISION}-wisdom-file ${WISDOM_FILE} \
--fftw-output-${PRECISION}-wisdom-file ${WISDOM_FILE} \
--fftw-threads-backend ${FFTW_THREADS_BACKEND} \
--fft-backends fftw --processing-scheme cpu:env >> $logfile
echo "'FFTW_PATIENT' planning" >> $logfile

echo "*****************************************************" >> $logfile
echo "                  Measure Level Zero        " >> $logfile
echo "****************************************************" >> $logfile

many_problem_bench --mbench-nthreads-env-name PYCBC_NUM_THREADS --mbench-wait-time ${MBWAIT} --mbench-cpu-affinity-list \
${CPULIST} --clear-cache \
--mbench-timing-program pycbc_timing --mbench-dummy-program pycbc_dummy \
--mbench-input-file ${INPUT_FILE} --mbench-output-file ${OUTPUT_FILE}-mlvl0.dat \
--mbench-input-arguments problem method --mbench-time ${MBTIME} --mbench-repeats ${MBREPEAT} \
--fftw-measure-level 0 --fftw-input-${PRECISION}-wisdom-file ${WISDOM_FILE} \
--fftw-threads-backend ${FFTW_THREADS_BACKEND} \
--fft-backends fftw --processing-scheme cpu:env >> $logfile
echo "'FFTW_PATIENT' (read from file) planning" >> $logfile

