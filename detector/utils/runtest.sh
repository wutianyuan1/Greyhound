export LD_PRELOAD=$LD_PRELOAD:/workspace/ncclprobe/build/libncclprobe.so
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMP_NUM_THREADS=1 mpirun -np 2 ../build/test