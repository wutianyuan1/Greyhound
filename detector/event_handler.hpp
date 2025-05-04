#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <numeric>
#include <sstream>
#include <nccl.h>
#include <cuda.h>
#include <cpp_redis/cpp_redis>
#include <boost/log/trivial.hpp>
#include "shm_topo.hpp"
#include "utils.hpp"

#define TASK_ACKED "TASK_ACKED"

enum ControlState : int32_t
{
    STATE_MONITOR  = 0,
    STATE_PROFILE  = 1,
    STATE_VALIDATE = 2,
};

enum ProcessRole : int32_t
{
    ROLE_SENDER = 0,
    ROLE_RECVER = 1,
};

struct ProfileResult
{
    double min_lat, max_lat, avg_lat, std_lat;
    ProfileResult() = default;
    ProfileResult(double minl, double maxl, double avgl, double stdl);
    std::string serialize();
};

class EventHandler
{
    ncclComm_t world_comm;
    Communicator parsed_comm;
    std::shared_ptr<cpp_redis::client> client;
    int* control_state;
    ncclSendFuncPtr send_ptr;
    ncclRecvFuncPtr recv_ptr;
public:
    EventHandler(std::string master_addr, int port, ncclSendFuncPtr sptr, ncclRecvFuncPtr rptr);
    ~EventHandler();
    bool has_world_comm() const;
    void set_world_comm(ncclComm_t comm);
    void fetech_and_exec_task();
    void handle_control_signal(cudaStream_t curr_stream, ControlState* state);
};
