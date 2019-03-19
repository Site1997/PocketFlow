#!/bin/bash

# configure pip3 to use internal source
unset http_proxy
unset https_proxy
mkdir -p ~/.pip3/ \
    && echo "[global]"                                              > ~/.pip3/pip3.conf \
    && echo "index-url = http://mirror-sng.oa.com/pypi/web/simple/" >> ~/.pip3/pip3.conf \
    && echo "trusted-host = mirror-sng.oa.com"                      >> ~/.pip3/pip3.conf
cat ~/.pip3/pip3.conf

# install python3 packages with Internet access
pip3 install tensorflow-gpu==1.12.0
pip3 install horovod
pip3 install docopt
pip3 install hdfs
pip3 install scipy
pip3 install sklearn
pip3 install pandas
pip3 install mpi4py

# add the current directory to python3PATH
export python3PATH=${python3PATH}:`pwd`
export LD_LIBRARY_PATH=/opt/ml/disk/local/cuda/lib64:$LD_LIBRARY_PATH

# start TensorBoard
LOG_DIR=/opt/ml/log
mkdir -p ${LOG_DIR}
nohup tensorboard \
    --port=${SEVEN_HTTP_FORWARD_PORT} \
    --host=127.0.0.1 \
    --logdir=${LOG_DIR} \
    >/dev/null 2>&1 &

# execute the main script
mkdir models
EXTRA_ARGS=`cat ./extra_args`
if [ ${NB_GPUS} -eq 1 ]; then
  echo "multi-GPU training disabled"
  python3 main.py --log_dir ${LOG_DIR} ${EXTRA_ARGS}
elif [ ${NB_GPUS} -le 8 ]; then
  echo "multi-GPU training enabled"
  options="-np ${NB_GPUS} -H localhost:${NB_GPUS} -bind-to none -map-by slot
      -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1
      -x LD_LIBRARY_PATH --mca btl_tcp_if_include eth1"
  mpirun ${options} python3 main.py --enbl_multi_gpu --log_dir ${LOG_DIR} ${EXTRA_ARGS}
fi

# archive model files to HDFS
mv models* /opt/ml/model

# remove *.pyc files
find . -name "*.pyc" -exec rm -f {} \;
