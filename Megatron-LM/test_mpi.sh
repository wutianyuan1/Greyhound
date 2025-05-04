# Usage: mpirun --allow-run-as-root --oversubscribe -np 8 bash ./test_mpi.sh
echo $OMPI_COMM_WORLD_RANK

# 1 MPI rank corresponds to 1 device
CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK python entrypoint.py $OMPI_COMM_WORLD_RANK

# 1 MPI rank corresponds to 2 devices
# D0=$(($OMPI_COMM_WORLD_RANK*2))
# D1=$(($OMPI_COMM_WORLD_RANK*2+1))
# CUDA_VISIBLE_DEVICES=$D0,$D1 python entrypoint.py $OMPI_COMM_WORLD_RANK

# 1 MPI rank corresponds to 4 devices
# D0=$(($OMPI_COMM_WORLD_RANK*4))
# D1=$(($OMPI_COMM_WORLD_RANK*4+1))
# D2=$(($OMPI_COMM_WORLD_RANK*4+2))
# D3=$(($OMPI_COMM_WORLD_RANK*4+3))
# CUDA_VISIBLE_DEVICES=$D0,$D1,$D2,$D3 python entrypoint.py $OMPI_COMM_WORLD_RANK