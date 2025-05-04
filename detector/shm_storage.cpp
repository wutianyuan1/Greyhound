#include <iostream>
#include <cassert>
#include <boost/log/trivial.hpp>
#include "shm_storage.hpp"
#include "utils.hpp"
using namespace boost::interprocess;


static void print_vec(std::vector<uint64_t>& v){
    std::cout << "Vector{";
    for (auto&& i:v) std::cout << i << ",";
    std::cout << "}\n";
}


/* === Buffer Implementation === */
RecordBuffer::RecordBuffer(size_t numFields, size_t maxRecords, void* mem_address)
    : numFields(numFields), maxRecords(maxRecords), numRecords(0), eventID(0), head(0), tail(0)
{
    this->addr = reinterpret_cast<uint64_t*>(mem_address);
    this->buffer = this->addr + METADATA_FIELDS;
    char init_msg[256];
    sprintf(init_msg, "[Before buffer init] Rank:%d, num:%lu, head:%lu, tail:%lu\n", get_rank(DistEngine::auto_find), addr[2], addr[3], addr[4]);
    BOOST_LOG_TRIVIAL(info) << init_msg;
    // addr[5] == BUFFER_MAGIC indicates it is initialized
    if (addr[5] != BUFFER_MAGIC)
    {
        addr[5] = BUFFER_MAGIC;
        updateMeta(true);
    }    
}

RecordBuffer::~RecordBuffer() {}

void RecordBuffer::loadMeta()
{
    numFields = addr[0];
    maxRecords = addr[1];
    numRecords = addr[2];
    head = addr[3];
    tail = addr[4];
    eventID = addr[6];
}

void RecordBuffer::updateMeta(bool init)
{
    addr[0] = numFields;
    addr[1] = maxRecords;
    addr[2] = numRecords;
    addr[3] = head;
    addr[4] = tail;
    if (!init)
        addr[6] = eventID + 1;
}

void RecordBuffer::push(std::vector<uint64_t>&& record)
{
    assert(record.size() == numFields);
    loadMeta();
    // printf("before RANK:%s, num:%lu, head:%lu, tail:%lu\n", getenv("RANK"), addr[2], addr[3], addr[4]);
    // HACK: store event ID in the last cell of record...
    record[record.size() - 1] = eventID;
    memcpy(buffer + tail * numFields, record.data(), numFields * sizeof(uint64_t));
    tail = (tail + 1) % maxRecords;
    if (numRecords == maxRecords)
        head = (head + 1) % maxRecords;
    if (numRecords < maxRecords)
        numRecords++;
    updateMeta(false);
    // printf("after RANK:%s, num:%lu, head:%lu, tail:%lu\n", getenv("RANK"), addr[2], addr[3], addr[4]);
}

void RecordBuffer::push(std::vector<uint64_t>& record)
{
    this->push(std::move(record));
}

inline bool RecordBuffer::full() noexcept
{
    return numRecords == maxRecords;
}

inline bool RecordBuffer::empty() noexcept
{
    return numRecords == 0;
}


/* === Storage Implementation === */
NcclRecordStorage::NcclRecordStorage(size_t numFields, size_t maxRecords)
    : numFields(numFields), maxRecords(maxRecords)
{
    shm = shared_memory_object(open_or_create, "ncclRecord", read_write);
    shm.truncate((METADATA_FIELDS + numFields * maxRecords) * sizeof(uint64_t));
    region = mapped_region(shm, read_write);
    lock_shm = shared_memory_object(open_or_create, "recordLock", read_write);
    lock_shm.truncate(sizeof(Lock));
    lock_region = mapped_region(lock_shm, read_write);
    lock = new (lock_region.get_address()) Lock;
    buffer = RecordBuffer(numFields, maxRecords, region.get_address());
}

NcclRecordStorage::~NcclRecordStorage()
{
    shared_memory_object::remove("ncclRecord");
    shared_memory_object::remove("recordLock");
    std::cout << "Shutdown gracefully!" << std::endl;
}

void NcclRecordStorage::addRecord(std::vector<uint64_t>&& record)
{
    scoped_lock<interprocess_mutex> slock(lock->mutex);
    buffer.push(std::move(record));
}

void NcclRecordStorage::addRecord(std::vector<uint64_t>& record)
{
    this->addRecord(std::move(record));
}
