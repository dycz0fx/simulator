#include "simulator.h"

// class Device
Device::Device(string name, DeviceType type)
: name(name), type(type)
{
}

// class Comp_device
Comp_device::Comp_device(string name)
: Device(name, Device::DEVICE_COMP)
{
}

// class Comm_device
Comm_device::Comm_device(string name, float latency, float bandwidth)
: Device(name, Device::DEVICE_COMM), latency(latency), bandwidth(bandwidth)
{
}

// class Task
Task::Task(string name, Device *device) 
: name(name), device(device), ready_time(0.0f), counter(0)
{
    next_tasks.clear();
} 

void Task::add_next_task(Task *task)
{
    next_tasks.push_back(task);
    task->counter++;
}

// class Comp_task
Comp_task::Comp_task(string name, Device *device, float run_time)
: Task(name, device), run_time(run_time)
{
}

float Comp_task::cost()
{
    return run_time;
}

// class Comm_task
Comm_task::Comm_task(string name, Device *device, int message_size)
: Task(name, device), message_size(message_size)
{
}

float Comm_task::cost()
{
    Comm_device *comm = (Comm_device *) device;
    return comm->latency + message_size / comm->bandwidth;
}

// class Simulator
Task *Simulator::new_comp_task(vector<Task *> prev_tasks, string name, Device *device, float run_time)
{
    Task *cur_task = (Task *) new Comp_task(name, device, run_time);
    if (prev_tasks.empty()) {
        ready_queue.push(cur_task);
    }
    else {
        for (size_t i = 0; i < prev_tasks.size(); i++) {
            prev_tasks[i]->add_next_task(cur_task);
        }
    }
    return cur_task;
}

vector<Task *> Simulator::new_comm_task(vector<Task *> prev_tasks, string name, Device *device, int message_size)
{
    vector<Task *> ret_tasks;
    // Divide messages into segments
    int num_segment = message_size / SEG_SIZE;
    int seg_size = SEG_SIZE;
    for (int i = 0; i < num_segment; i++) {
        if (i == num_segment - 1) {
            seg_size = message_size - (num_segment - 1) * SEG_SIZE;
        }
        Task *cur_task = (Task *) new Comm_task(name, device, seg_size);
        if (prev_tasks.empty()) {
            ready_queue.push(cur_task);
        }
        else {
            for (size_t i = 0; i < prev_tasks.size(); i++) {
                prev_tasks[i]->add_next_task(cur_task);
            }        
        }
        ret_tasks.push_back(cur_task);
    }
    return ret_tasks;
}

void Simulator::simulate()
{
    srand (time(NULL));
    float sim_time = 0.0f;
    unordered_map<Device*, float> device_times;
    while (!ready_queue.empty()) {
        // Find the task with the earliest start time
        Task* cur_task = ready_queue.top();
        ready_queue.pop();
        float ready_time = 0;
        if (device_times.find((Device *)cur_task->device) != device_times.end()) {
            ready_time = device_times[(Device *)cur_task->device];
        }
        float start_time = std::max(ready_time, cur_task->ready_time);
        float run_time = 0;
        if (cur_task->device->type == Device::DEVICE_COMP) {
            run_time = ((Comp_task *)cur_task)->cost();
        }
        else {
            run_time = ((Comm_task *)cur_task)->cost();
        }
        float end_time = start_time + run_time;
        device_times[cur_task->device] = end_time;
        cout << cur_task->name << " on " << cur_task->device->name << " --- " << "ready(" << 
            ready_time << ") start("  << start_time << ") run(" << run_time << ") end(" <<  end_time << ")" << endl;
        if (end_time > sim_time)
            sim_time = end_time;
        for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
            Task* next = cur_task->next_tasks[i];
            next->ready_time = max(next->ready_time, end_time);
            next->counter--;
            if (next->counter == 0) {
                ready_queue.push(next);
            }
        }
    }

    cout << "sim_time " << sim_time << endl;
    return;
}

int main()
{
    int size = 4;
    vector<Comp_device> cpus;
    for (int i = 0; i < size; i++){
        string cpu_name = "CPU" + to_string(i);
        cpus.emplace_back(Comp_device(cpu_name));
    }
    Comm_device dram("DRAM", 0, 1); // ms, B/ms

    Simulator simulator;
    // 0 1 2 3
    //  / / /
    // 0 1 2 3
    // 0st iteration
    vector<Task *> comp_tasks_0;
    for (int i = 0; i < size; i++){
        string task_name = "[0] compute " + to_string(i);
        int run_time = 10;
        comp_tasks_0.emplace_back(simulator.new_comp_task({}, task_name, (Device *)(&cpus[i]), run_time));
    }

    // communication
    unordered_map<int, vector<Task *>> comm_tasks_0;
    for (int i = 0; i < size; i++){
        int left = i - 1;
        if (left >= 0) {
            string task_name = "[0] send from " + to_string(i) + " to " + to_string(left);
            vector<Task *> prev_tasks;
            prev_tasks.emplace_back(comp_tasks_0[i]);
            vector<Task *> comm_tasks = simulator.new_comm_task(prev_tasks, task_name, (Device *)&dram, 10);
            comm_tasks_0[left].insert(comm_tasks_0[left].end(), comm_tasks.begin(), comm_tasks.end());
        }
        int right = i + 1;
        if (right < size) {
            string task_name = "[0] send from " + to_string(i) + " to " + to_string(right);
            vector<Task *> prev_tasks;
            prev_tasks.emplace_back(comp_tasks_0[i]);
            vector<Task *> comm_tasks = simulator.new_comm_task(prev_tasks, task_name, (Device *)&dram, 10);
            comm_tasks_0[right].insert(comm_tasks_0[right].end(), comm_tasks.begin(), comm_tasks.end());
        }
    }

    // 1st iteration
    vector<Task *> comp_tasks_1;
    for (int i = 0; i < size; i++){
        string task_name = "[1] compute " + to_string(i);
        vector<Task *> prev_tasks;
        prev_tasks.insert(prev_tasks.end(), comm_tasks_0[i].begin(), comm_tasks_0[i].end());
        comp_tasks_1.emplace_back(simulator.new_comp_task(prev_tasks, task_name, (Device *)(&cpus[i]), 10));
    }

    simulator.simulate();

}


