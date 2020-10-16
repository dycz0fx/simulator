#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <time.h>

#define SEG_SIZE 4096
#define MAX_NUM_SEGS 1000
#define LEGION_OVERHEAD 9
class Device
{
public:
    enum DeviceType {
        DEVICE_COMP,
        DEVICE_COMM,
    };
    Device(std::string name, DeviceType type, int node_id, int socket_id, int device_id);
    std::string name;
    DeviceType type;
    int node_id;
    int socket_id;
    int device_id;
};

class Comp_device : public Device
{
public:
    Comp_device(std::string name, int node_id, int socket_id, int device_id);
};

class Comm_device : public Device
{
public:
    float latency;
    float bandwidth;
    Comm_device(std::string name, int node_id, int socket_id, int device_id, float latency, float bandwidth);
};

class Machine
{
private:
    std::unordered_map<int, Comm_device *> comp_to_dram;
    std::unordered_map<int, Comm_device *> comp_to_qpi_in;
    std::unordered_map<int, Comm_device *> comp_to_qpi_out;
    std::unordered_map<int, Comm_device *> comp_to_nic_in;
    std::unordered_map<int, Comm_device *> comp_to_nic_out;
public:
    Machine();
    void attach_dram(Comp_device *comp, Comm_device *comm);
    void attach_qpi_in(Comp_device *comp, Comm_device *comm);
    void attach_qpi_out(Comp_device *comp, Comm_device *comm);
    void attach_nic_in(Comp_device *comp, Comm_device *comm);
    void attach_nic_out(Comp_device *comp, Comm_device *comm);
    std::vector<Comm_device *> get_comm_path(Comp_device *source, Comp_device *target);
};

class Task
{
public:
    Task(std::string name, Device *device);
    std::string name;
    Device *device;
    float ready_time;
    std::vector<Task *> next_tasks;
    int counter;
    void add_next_task(Task *task);
    virtual std::string to_string();
};

class Comp_task : public Task
{
public:
    Comp_task(std::string name, Comp_device *device, float run_time);
    float run_time;
    float cost();
    std::string to_string();
};

class Comm_task : public Task
{
public:
    Comm_task(std::string name, Comm_device *device, int message_size);
    int message_size;
    float cost();
    std::string to_string();
};

class TaskCompare {
public:
    bool operator() (Task *lhs, Task *rhs) {
        if (lhs->ready_time == rhs->ready_time) {
            return rand() % 2;
        }
        return lhs->ready_time > rhs->ready_time;
    }
};

class Simulator
{
private:
    std::priority_queue<Task *, std::vector<Task *>, TaskCompare> ready_queue;
public:
    Machine machine;
    Task *new_comp_task(std::string name, Comp_device *device, float run_time);
    void new_comm_task(Task *source_task, Task *target_task, int message_size);
    void enter_ready_queue(Task *task);
    void add_dependency(std::vector<Task *> prev_tasks, Task *cur_task);
    void add_dependency(Task *prev_task, Task *cur_task);
    void simulate();
};

bool starts_with(std::string s, std::string sub){
    return s.find(sub) == 0 ? 1 : 0;
}
