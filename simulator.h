#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <time.h>

using namespace std;

#define SEG_SIZE 5

class Device
{
public:
    enum DeviceType {
        DEVICE_COMP,
        DEVICE_COMM,
    };
    Device(string name, DeviceType type);
    string name;
    DeviceType type;
};

class Comp_device : public Device
{
public:
    Comp_device(string name);
};

class Comm_device : public Device
{
public:
    float latency;
    float bandwidth;
    Comm_device(string name, float latency, float bandwidth);
};

class Task
{
private:
    /* data */
public:
    Task(string name, Device *device);
    string name;
    Device *device;
    float ready_time;
    vector<Task *> next_tasks;
    int counter;
    void add_next_task(Task *task);
};

class Comp_task : public Task
{
public:
    Comp_task(string name, Device *device, float run_time);
    float run_time;
    float cost();
};

class Comm_task : public Task
{
public:
    Comm_task(string name, Device *device, int message_size);
    int message_size;
    float cost();
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
    priority_queue<Task *, vector<Task *>, TaskCompare> ready_queue;
public:
    Task *new_comp_task(vector<Task *> prev_tasks, string name, Device *device, float run_time);
    vector<Task *> new_comm_task(vector<Task *> prev_tasks, string name, Device *device, int message_size);
    void simulate();
};
