#pragma once
#include <vector>
#include <atomic>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "config.hpp"


struct Record
{
    uint64_t commAddr, callNumber, count, buff1Addr, buff2Addr, datatype, pid,
        callTime, deviceId, caller, aux, duration, numDevices, event_id;

    Record(uint64_t commAddr_, uint64_t callNumber_, uint64_t count_,
           uint64_t buff1Addr_, uint64_t buff2Addr_,
           uint64_t datatype_, uint64_t pid_, uint64_t callTime_,
           uint64_t deviceId_, uint64_t caller_, uint64_t aux_,
           uint64_t duration_, uint64_t numdevs)
      : commAddr(commAddr_), callNumber(callNumber_), count(count_), buff1Addr(buff1Addr_),
        buff2Addr(buff2Addr_), datatype(datatype_), pid(pid_), callTime(callTime_),
        deviceId(deviceId_), caller(caller_), aux(aux_),
        duration(duration_), numDevices(numdevs), event_id(0)
    {}

    std::vector<uint64_t> toVector()
    {
        std::vector<uint64_t> res({commAddr, callNumber, count, buff1Addr, buff2Addr, datatype,
            pid, callTime, deviceId, caller, aux, duration, numDevices, event_id});
        return res;
    }

    static constexpr size_t numFields()
    {
        return sizeof(struct Record) / sizeof(uint64_t);
    }
};


class RecordBuffer
{
    size_t numFields;
    size_t maxRecords;
    size_t numRecords;
    size_t eventID;
    off_t head;
    off_t tail;
    uint64_t* buffer;
    uint64_t* addr;
public:
    RecordBuffer() = default;
    RecordBuffer(size_t numFields, size_t maxRecords, void* mem_address);
    ~RecordBuffer();
    void push(std::vector<uint64_t>&& record);
    void push(std::vector<uint64_t>& record);
    bool full() noexcept;
    bool empty() noexcept;
private:
    void loadMeta();
    void updateMeta(bool init);
};


struct Lock
{
    boost::interprocess::interprocess_mutex  mutex;
};


class NcclRecordStorage
{
    boost::interprocess::shared_memory_object shm;
    boost::interprocess::mapped_region region;
    boost::interprocess::shared_memory_object lock_shm;
    boost::interprocess::mapped_region lock_region;
    Lock* lock;
    size_t numFields;
    size_t maxRecords;
    RecordBuffer buffer;
public:
    NcclRecordStorage() = default;
    NcclRecordStorage(size_t numFields, size_t maxRecords);
    ~NcclRecordStorage();
    void addRecord(std::vector<uint64_t>&& record);
    void addRecord(std::vector<uint64_t>& record);
};
