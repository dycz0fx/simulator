#include "simulator.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::unordered_map;
using std::max;

// class Device
Device::Device(string name, DeviceType type, int id)
: name(name), type(type), id(id)
{
}

// class Comp_device
Comp_device::Comp_device(string name, int id)
: Device(name, Device::DEVICE_COMP, id)
{
}

// class Comm_device
Comm_device::Comm_device(string name, int id, float latency, float bandwidth)
: Device(name, Device::DEVICE_COMM, id), latency(latency), bandwidth(bandwidth)
{
}

// class Machine
Machine::Machine() {
    comp_to_dram.clear();
    comp_to_nic_in.clear();
    comp_to_nic_out.clear();
}

void Machine::attach_dram(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "DRAM"));
    if (comp_to_dram.find(comp->id) == comp_to_dram.end()) {
        comp_to_dram[comp->id] = comm;
    }
}

void Machine::attach_nic_in(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "NIC"));
    if (comp_to_nic_in.find(comp->id) == comp_to_nic_in.end()) {
        comp_to_nic_in[comp->id] = comm;
    }
}

void Machine::attach_nic_out(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "NIC"));
    if (comp_to_nic_out.find(comp->id) == comp_to_nic_out.end()) {
        comp_to_nic_out[comp->id] = comm;
    }
}

vector<Comm_device *> Machine::get_comm_path(Comp_device *source, Comp_device *target)
{
    vector<Comm_device *> ret;
    // on the same node
    if (source->id == target->id) {
        ret.emplace_back(comp_to_dram[source->id]);
    }
    // on different nodes
    else {
        ret.emplace_back(comp_to_dram[source->id]);
        ret.emplace_back(comp_to_nic_out[source->id]);
        ret.emplace_back(comp_to_nic_in[target->id]);
        ret.emplace_back(comp_to_dram[target->id]);
    }
    return ret;
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
Comp_task::Comp_task(string name, Comp_device *device, float run_time)
: Task(name, device), run_time(run_time)
{
}

float Comp_task::cost()
{
    return run_time;
}

// class Comm_task
Comm_task::Comm_task(string name, Comm_device *device, int message_size)
: Task(name, device), message_size(message_size)
{
}

float Comm_task::cost()
{
    Comm_device *comm = (Comm_device *) device;
    return comm->latency + message_size / comm->bandwidth;
}

// class Simulator
Task *Simulator::new_comp_task(string name, Comp_device *device, float run_time)
{
    Task *cur_task = (Task *) new Comp_task(name, device, run_time);
    return cur_task;
}

std::vector<Task *> Simulator::new_comm_task(Task *source_task, Task *target_task, int message_size)
{
    vector<Comm_device *> path = machine.get_comm_path((Comp_device *)(source_task->device), (Comp_device *)(target_task->device));
    vector<vector<Task *>> all_tasks;
    int num_segment = message_size / SEG_SIZE;
    // Create all the comm tasks
    // Divide messages into segments
    for (int i = 0; i < path.size(); i++) {
        all_tasks.push_back({});
        for (int j = 0; j < num_segment; j++) {
            int seg_size = SEG_SIZE;
            if (j == num_segment - 1) {
                seg_size = message_size - (num_segment - 1) * SEG_SIZE;
            }
            string name = "seg " + to_string(j) + " from " + source_task->name + " to " + target_task->name;
            Task *cur_task = (Task *) new Comm_task(name, path[i], seg_size);
            all_tasks[i].push_back(cur_task);
        }
    }

    // Add dependencies among the comm tasks
    for (int i = 0; i < path.size(); i++) {
        for (int j = 0; j < num_segment; j++) {
            if (i == 0) {
                add_dependency(source_task, all_tasks[i][j]);
            }
            if (i == path.size() - 1) {
                add_dependency(all_tasks[i][j], target_task);
            }
            if (i > 0) {
                add_dependency(all_tasks[i-1][j], all_tasks[i][j]);
            }
        }
    }

    return all_tasks.back();
}

void Simulator::enter_ready_queue(Task *task)
{
    ready_queue.push(task);
}

void Simulator::add_dependency(vector<Task *> prev_tasks, Task *cur_task)
{
    for (int i = 0; i < prev_tasks.size(); i++) {
        prev_tasks[i]->add_next_task(cur_task);
    }
    
}

void Simulator::add_dependency(Task *prev_task, Task *cur_task)
{
    prev_task->add_next_task(cur_task);
}

void Simulator::simulate()
{
    srand(time(NULL));
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
        cout << cur_task->name << " --- " << cur_task->device->name << " --- " << "ready(" << 
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
    Simulator simulator;

    int num_tasks = 4;
    int num_nodes = 2;
    int num_tasks_per_node = num_tasks / num_nodes;

    vector<vector<Comp_device *>> cpus;
    for (int i = 0; i < num_nodes; i++) {
        cpus.push_back({});
        for (int j = 0; j < num_tasks_per_node; j++) {
            string cpu_name = "CPU " + to_string(j) + " on NODE " + to_string(i);
            cpus[i].emplace_back(new Comp_device(cpu_name, i));
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        string dram_name = "DRAM " + to_string(i) + " on NODE " + to_string(i);
        Comm_device *dram = new Comm_device(dram_name, i, 0, 1); // ms, B/ms or kB/s
        // drams.emplace_back(Comm_device(dram_name, i, 0.00003, 8.75 * 1024 * 1024)); // ms, B/ms or kB/s
        string nic_in_name = "NIC_IN " + to_string(i) + " on NODE " + to_string(i);
        Comm_device *nic_in = new Comm_device(nic_in_name, i, 0, 2);
        // nics_in.emplace_back(Comm_device(nic_in_name, i, 0.006214, 5 * 1024 * 1024));
        string nic_out_name = "NIC_OUT " + to_string(i) + " on NODE " + to_string(i);
        Comm_device *nic_out = new Comm_device(nic_out_name, i, 0, 2);
        // nics_out.emplace_back(Comm_device(nic_out_name, i, 0.006214, 5 * 1024 * 1024));
        simulator.machine.attach_dram(cpus[i][0], dram);
        simulator.machine.attach_nic_in(cpus[i][0], nic_in);
        simulator.machine.attach_nic_out(cpus[i][0], nic_out);
    }

    // 1D stencil
    // 0: 0 1
    // 1: 0 1 2
    // 2: 1 2 3
    // 3: 2 3

    // init comp tasks first
    // iteration 0
    vector<Task *> comp_tasks_0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_tasks_per_node; j++) {
            int k = i * num_tasks_per_node + j;
            string task_name = "[0] compute " + to_string(k);
            int run_time = 10;
            comp_tasks_0.emplace_back(simulator.new_comp_task(task_name, cpus[i][j], run_time));
        }
    }

    for (int i = 0; i < comp_tasks_0.size(); i++) {
        simulator.enter_ready_queue((Task *)comp_tasks_0[i]);
    }

    // iteration 1
    vector<Task *> comp_tasks_1;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_tasks_per_node; j++) {
            int k = i * num_tasks_per_node + j;        
            string task_name = "[1] compute " + to_string(k);
            int run_time = 10;
            comp_tasks_1.emplace_back(simulator.new_comp_task(task_name, cpus[i][j], run_time));
        }
    }

    // add comm tasks between iter 0 and iter 1
    vector<Task *> comm_tasks_0;
    for (int i = 0; i < num_tasks; i++){
        int message_size = 10;
        // left
        int left = i - 1;
        if (left >= 0) {
            vector<Task *> comm_tasks = simulator.new_comm_task(comp_tasks_0[i], comp_tasks_1[left], 10);
            comm_tasks_0.insert(comm_tasks_0.end(), comm_tasks.begin(), comm_tasks.end());
        }
        /*
        // right
        int right = i + 1;
        if (right < num_tasks) {
            vector<Task *> comm_tasks = simulator.new_comm_task(comp_tasks_0[i], comp_tasks_1[right], 10);
            comm_tasks_0.insert(comm_tasks_0.end(), comm_tasks.begin(), comm_tasks.end());
        }
        */
        // mid
        vector<Task *> comm_tasks = simulator.new_comm_task(comp_tasks_0[i], comp_tasks_1[i], 10);
        comm_tasks_0.insert(comm_tasks_0.end(), comm_tasks.begin(), comm_tasks.end());
    }

    simulator.simulate();

}


