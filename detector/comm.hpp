#pragma once
#include <nccl.h>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <memory>
#include "shm_topo.hpp"
#include "config.hpp"

#define COMM_RANK_UNDEFINED INT32_MIN

/* Parses the opaque NCCL communicator via reverse engineering, save
   the parsed information to *parsed_comm */
void parse_communicator(ncclComm_t hidden_comm, Communicator* parsed_comm);
