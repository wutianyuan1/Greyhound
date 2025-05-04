#pragma once
#include <string>
#include <cmath>
#include <nccl.h>

// floating point comparison
#define FLOAT_EQ(a, b) (fabs((a) - (b)) < 1e-4)

enum DistEngine{
    undefined = 0,
    mpi = 1,
    torch_run = 2,
    auto_find = 3
};

typedef ncclResult_t (*ncclSendFuncPtr)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t (*ncclRecvFuncPtr)(const void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);


inline int get_int_value_from_env(DistEngine dist, const char* mpi_choice, const char* torchrun_choice)
{
    if (dist == DistEngine::mpi)
        return std::atoi(getenv(mpi_choice));
    else if (dist == DistEngine::torch_run)
        return std::atoi(getenv(torchrun_choice));
    else if (dist == DistEngine::auto_find)
        return getenv(mpi_choice) ?\
            std::atoi(getenv(mpi_choice)) : std::atoi(getenv(torchrun_choice));
    else  // default: "torchrun_choice" from env
        return std::atoi(getenv(torchrun_choice));
}


inline int get_rank(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_RANK", "RANK");
}

inline int get_local_rank(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK");
}

inline int get_world_size(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_SIZE", "WORLD_SIZE");
}

inline int get_local_world_size(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_LOCAL_SIZE", "LOCAL_WORLD_SIZE");
}

inline const char* get_nccl_path(const char* nccl_path = nullptr)
{
    return getenv(nccl_path ? nccl_path : "NCCL_PATH");
}

inline const char* get_master_addr()
{
    char* maddr = getenv("MASTER_ADDR");
    return maddr ? maddr : "127.0.0.1"; 
}

inline int get_redis_port()
{
    char* rport = getenv("REDIS_PORT");
    return rport ? std::atoi(rport) : 6379; 
}

inline const char* get_whl_path()
{
    char* whl_path = getenv("CONTROL_PLANE_WHL_PATH");
    return whl_path ? whl_path : "/workspace/ncclprobe/dist/control_plane-1.0-py3-none-any.whl";
}

inline const char* get_local_controller_log_path()
{
    char* whl_path = getenv("LOCAL_CONTROLLER_LOG_PATH");
    return whl_path;
}

inline const char* get_global_controller_log_path()
{
    char* whl_path = getenv("GLOBAL_CONTROLLER_LOG_PATH");
    return whl_path;
}

inline const char* get_probe_log_path()
{
    char* whl_path = getenv("NCCLPROBE_LOG_PATH");
    return whl_path;
}