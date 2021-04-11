#include "simulator.h"
#include <unordered_set>
#include <fstream>
#include <utility>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::unordered_map;
using std::unordered_set;
using std::pair;

vector<string> split(string srcStr, const string& delim)
{
	int nPos = 0;
	vector<string> vec;
	nPos = srcStr.find(delim.c_str());
	while(-1 != nPos)
	{
		string temp = srcStr.substr(0, nPos);
		vec.push_back(temp);
		srcStr = srcStr.substr(nPos+1);
		nPos = srcStr.find(delim.c_str());
	}
	vec.push_back(srcStr);
	return vec;
}

void stencil_1d_cpu()
{
    int num_nodes = 2;
    int num_sockets_per_node = 2;
    int num_cpus_per_socket = 2;
    int num_gpus_per_socket = 0;
    int num_sockets = num_nodes * num_sockets_per_node;
    int num_cpus = num_sockets * num_cpus_per_socket;

    EnhancedMachineModel machine("/Users/xluo/Documents/simulator/final/machine_config");
    
    Simulator simulator(&machine);

    // 1D stencil
    // 0: 0 1
    // 1: 0 1 2
    // 2: 1 2 3
    // 3: 2 3

    // init comp tasks first
    int iters = 2;
    vector<vector <Task *> > comp_tasks;
    for (int t = 0; t < iters; t++) {
        comp_tasks.push_back({});
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_sockets_per_node; j++) {
                for (int k = 0; k < num_cpus_per_socket; k++) {
                    int socket_id = i * num_sockets_per_node + j;
                    int task_id = socket_id * num_cpus_per_socket + k;
                    string task_name = "comp_task " + to_string(task_id) + " iter " + to_string(t);
                    float run_time = 1;
                    comp_tasks[t].emplace_back(simulator.new_comp_task(task_name, machine.get_cpu(socket_id, k), run_time, machine.get_sys_mem(socket_id)));
                }
            }
        }
    }

    for (int i = 0; i < comp_tasks[0].size(); i++) {
        simulator.enter_ready_queue((Task *)comp_tasks[0][i]);
    }

    // add comm tasks between iters
    for (int i = 1; i < iters; i++) {
        for (int j = 0; j < num_cpus; j++){
            long message_size = 262144;
            // left
            int left = j - 1;
            if (left >= 0) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][left], message_size);
            }
            // right
            int right = j + 1;
            if (right < num_cpus) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][right], message_size);
            }
            // mid
            simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][j], message_size);
        }
    }

    simulator.simulate();
}

void stencil_1d_gpu()
{
    int num_nodes = 2;
    int num_sockets_per_node = 1;
    int num_cpus_per_socket = 10;
    int num_gpus_per_socket = 4;
    int num_sockets = num_nodes * num_sockets_per_node;
    int num_cpus = num_sockets * num_cpus_per_socket;

    // EnhancedMachineModel machine("/Users/xluo/Documents/simulator/final/machine_config");
    
    SimpleMachineModel machine(num_nodes, num_sockets_per_node * num_cpus_per_socket, num_sockets_per_node * num_gpus_per_socket);
    Simulator simulator(&machine);

    // init comp tasks first
    int iters = 2;
    vector<vector <Task *>> comp_tasks;
    for (int t = 0; t < iters; t++) {
        comp_tasks.push_back({});
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_sockets_per_node; j++) {
                for (int k = 0; k < num_gpus_per_socket; k++) {
                    int socket_id = i * num_sockets_per_node + j;
                    int task_id = socket_id * num_gpus_per_socket + k;
                    string task_name = "comp_task " + to_string(task_id) + " iter " + to_string(t);
                    float run_time = 1;
                    comp_tasks[t].emplace_back(simulator.new_comp_task(task_name, machine.get_gpu(task_id), run_time, machine.get_gpu_fb_mem(task_id)));
                }
            }
        }
    }

    for (int i = 0; i < comp_tasks[0].size(); i++) {
        simulator.enter_ready_queue((Task *)comp_tasks[0][i]);
    }

    // add comm tasks between iters
    int num_tasks = num_nodes * num_sockets_per_node * num_gpus_per_socket;
    for (int i = 1; i < iters; i++) {
        for (int j = 0; j < num_tasks; j++){
            long message_size = 262144;
            // left
            int left = j - 1;
            if (left >= 0) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][left], message_size);
            }
            // right
            int right = j + 1;
            if (right < num_tasks) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][right], message_size);
            }
            // mid
            simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][j], message_size);
        }
    }

    simulator.simulate();

}

vector<vector<string>> utils_ids;
std::unordered_map<std::string, pair<int, int>> cpu_id_map;
std::unordered_map<std::string, pair<int, int>> gpu_id_map;
std::unordered_map<std::string, int> mem_id_map;

// TODO: init id maps automaticly
static void init_id_maps()
{
    /* for alexnet, resnet, inception */
    utils_ids.push_back({});
    utils_ids.back().push_back("0x1d00000000000000");
    utils_ids.back().push_back("0x1d00000000000001");
    utils_ids.push_back({});
    utils_ids.back().push_back("0x1d00010000000000");
    utils_ids.back().push_back("0x1d00010000000001");

    // set up cpu_id_map: device_id
    cpu_id_map["0x1d00000000000000"] = {0, 0};
    cpu_id_map["0x1d00000000000001"] = {0, 1};
    cpu_id_map["0x1d00000000000002"] = {0, 2};

    cpu_id_map["0x1d00010000000000"] = {1, 20};
    cpu_id_map["0x1d00010000000001"] = {1, 21};
    cpu_id_map["0x1d00010000000002"] = {1, 22};

    // set up mem_id_map for system mem and zero copy mem : socket_id
    mem_id_map["0x1e00000000000000"] = {0};
    mem_id_map["0x1e00000000000005"] = {0};
    mem_id_map["0x1e00010000000000"] = {1};
    mem_id_map["0x1e00010000000005"] = {1};
    // set up gpu_id_map: device_id
    gpu_id_map["0x1d00000000000003"] = {0, 0};
    gpu_id_map["0x1d00000000000004"] = {0, 1};
    gpu_id_map["0x1d00000000000005"] = {0, 2};
    gpu_id_map["0x1d00000000000006"] = {0, 3};
    gpu_id_map["0x1d00010000000003"] = {1, 4};
    gpu_id_map["0x1d00010000000004"] = {1, 5};
    gpu_id_map["0x1d00010000000005"] = {1, 6};
    gpu_id_map["0x1d00010000000006"] = {1, 7};
    mem_id_map["0x1e00000000000001"] = {0};
    mem_id_map["0x1e00000000000002"] = {1};
    mem_id_map["0x1e00000000000003"] = {2};
    mem_id_map["0x1e00000000000004"] = {3};
    mem_id_map["0x1e00010000000001"] = {4};
    mem_id_map["0x1e00010000000002"] = {5};
    mem_id_map["0x1e00010000000003"] = {6};
    mem_id_map["0x1e00010000000004"] = {7};

    /* for dlrm only */
    // utils_ids.push_back({});
    // utils_ids.back().push_back("0x1d00000000000000");
    // utils_ids.back().push_back("0x1d00000000000001");
    // utils_ids.back().push_back("0x1d00000000000002");
    // utils_ids.back().push_back("0x1d00000000000003");
    // utils_ids.push_back({});
    // utils_ids.back().push_back("0x1d00010000000000");
    // utils_ids.back().push_back("0x1d00010000000001");
    // utils_ids.back().push_back("0x1d00010000000002");
    // utils_ids.back().push_back("0x1d00010000000003");
    // // set up cpu_id_map: device_id
    // cpu_id_map["0x1d00000000000000"] = {0, 0};
    // cpu_id_map["0x1d00000000000001"] = {0, 1};
    // cpu_id_map["0x1d00000000000002"] = {0, 2};
    // cpu_id_map["0x1d00000000000003"] = {0, 3};
    // cpu_id_map["0x1d00000000000004"] = {0, 4};
    // cpu_id_map["0x1d00010000000000"] = {1, 20};
    // cpu_id_map["0x1d00010000000001"] = {1, 21};
    // cpu_id_map["0x1d00010000000002"] = {1, 22};
    // cpu_id_map["0x1d00010000000003"] = {1, 23};
    // cpu_id_map["0x1d00010000000004"] = {1, 24};
    // // set up mem_id_map for system mem and zero copy mem : socket_id
    // mem_id_map["0x1e00000000000000"] = {0};
    // mem_id_map["0x1e00000000000005"] = {0};
    // mem_id_map["0x1e00010000000000"] = {1};
    // mem_id_map["0x1e00010000000005"] = {1};
    // // set up gpu_id_map: device_id
    // gpu_id_map["0x1d00000000000005"] = {0, 0};
    // gpu_id_map["0x1d00000000000006"] = {0, 1};
    // gpu_id_map["0x1d00000000000007"] = {0, 2};
    // gpu_id_map["0x1d00000000000008"] = {0, 3};
    // gpu_id_map["0x1d00010000000005"] = {1, 4};
    // gpu_id_map["0x1d00010000000006"] = {1, 5};
    // gpu_id_map["0x1d00010000000007"] = {1, 6};
    // gpu_id_map["0x1d00010000000008"] = {1, 7};
    // mem_id_map["0x1e00000000000001"] = {0};
    // mem_id_map["0x1e00000000000002"] = {1};
    // mem_id_map["0x1e00000000000003"] = {2};
    // mem_id_map["0x1e00000000000004"] = {3};
    // mem_id_map["0x1e00010000000001"] = {4};
    // mem_id_map["0x1e00010000000002"] = {5};
    // mem_id_map["0x1e00010000000003"] = {6};
    // mem_id_map["0x1e00010000000004"] = {7};

    /* for circuit, stencil, pennant */
    // // set up util cores: socket_id, device_id
    // utils_ids.push_back({});
    // utils_ids.back().push_back("0x1d00000000000000");
    // utils_ids.back().push_back("0x1d00000000000001");
    // utils_ids.back().push_back("0x1d00000000000002");
    // utils_ids.back().push_back("0x1d00000000000003");
    // utils_ids.push_back({});
    // utils_ids.back().push_back("0x1d00010000000000");
    // utils_ids.back().push_back("0x1d00010000000001");
    // utils_ids.back().push_back("0x1d00010000000002");
    // utils_ids.back().push_back("0x1d00010000000003");
    // utils_ids.push_back({});
    // utils_ids.back().push_back("0x1d00020000000000");
    // utils_ids.back().push_back("0x1d00020000000001");
    // utils_ids.back().push_back("0x1d00020000000002");
    // utils_ids.back().push_back("0x1d00020000000003");
    // utils_ids.push_back({});
    // utils_ids.back().push_back("0x1d00030000000000");
    // utils_ids.back().push_back("0x1d00030000000001");
    // utils_ids.back().push_back("0x1d00030000000002");
    // utils_ids.back().push_back("0x1d00030000000003");
    // // set up cpu_id_map: device_id
    // cpu_id_map["0x1d00000000000000"] = {0, 0};
    // cpu_id_map["0x1d00000000000001"] = {0, 1};
    // cpu_id_map["0x1d00000000000002"] = {0, 2};
    // cpu_id_map["0x1d00000000000003"] = {0, 3};
    // cpu_id_map["0x1d00000000000004"] = {0, 4};
    // cpu_id_map["0x1d00000000000005"] = {0, 5};
    // cpu_id_map["0x1d00000000000006"] = {0, 6};
    // cpu_id_map["0x1d00000000000007"] = {0, 7};
    // cpu_id_map["0x1d00000000000008"] = {0, 8};
    // cpu_id_map["0x1d00000000000009"] = {0, 9};
    // cpu_id_map["0x1d0000000000000a"] = {0, 10};
    // cpu_id_map["0x1d0000000000000b"] = {0, 11};
    // cpu_id_map["0x1d0000000000000c"] = {0, 12};
    // cpu_id_map["0x1d0000000000000d"] = {0, 13};
    // cpu_id_map["0x1d0000000000000e"] = {0, 14};
    // cpu_id_map["0x1d0000000000000f"] = {0, 15};
    // cpu_id_map["0x1d00000000000010"] = {0, 16};
    // cpu_id_map["0x1d00000000000011"] = {0, 17};
    // cpu_id_map["0x1d00000000000012"] = {0, 18};
    // cpu_id_map["0x1d00000000000013"] = {0, 19};

    // cpu_id_map["0x1d00010000000000"] = {1, 20};
    // cpu_id_map["0x1d00010000000001"] = {1, 21};
    // cpu_id_map["0x1d00010000000002"] = {1, 22};
    // cpu_id_map["0x1d00010000000003"] = {1, 23};
    // cpu_id_map["0x1d00010000000004"] = {1, 24};
    // cpu_id_map["0x1d00010000000005"] = {1, 25};
    // cpu_id_map["0x1d00010000000006"] = {1, 26};
    // cpu_id_map["0x1d00010000000007"] = {1, 27};
    // cpu_id_map["0x1d00010000000008"] = {1, 28};
    // cpu_id_map["0x1d00010000000009"] = {1, 29};
    // cpu_id_map["0x1d0001000000000a"] = {1, 30};
    // cpu_id_map["0x1d0001000000000b"] = {1, 31};
    // cpu_id_map["0x1d0001000000000c"] = {1, 32};
    // cpu_id_map["0x1d0001000000000d"] = {1, 33};
    // cpu_id_map["0x1d0001000000000e"] = {1, 34};
    // cpu_id_map["0x1d0001000000000f"] = {1, 35};
    // cpu_id_map["0x1d00010000000010"] = {1, 36};
    // cpu_id_map["0x1d00010000000011"] = {1, 37};
    // cpu_id_map["0x1d00010000000012"] = {1, 38};
    // cpu_id_map["0x1d00010000000013"] = {1, 39};

    // cpu_id_map["0x1d00020000000000"] = {2, 40};
    // cpu_id_map["0x1d00020000000001"] = {2, 41};
    // cpu_id_map["0x1d00020000000002"] = {2, 42};
    // cpu_id_map["0x1d00020000000003"] = {2, 43};
    // cpu_id_map["0x1d00020000000004"] = {2, 44};
    // cpu_id_map["0x1d00020000000005"] = {2, 45};
    // cpu_id_map["0x1d00020000000006"] = {2, 46};
    // cpu_id_map["0x1d00020000000007"] = {2, 47};
    // cpu_id_map["0x1d00020000000008"] = {2, 48};
    // cpu_id_map["0x1d00020000000009"] = {2, 49};
    // cpu_id_map["0x1d0002000000000a"] = {2, 50};
    // cpu_id_map["0x1d0002000000000b"] = {2, 51};
    // cpu_id_map["0x1d0002000000000c"] = {2, 52};
    // cpu_id_map["0x1d0002000000000d"] = {2, 53};
    // cpu_id_map["0x1d0002000000000e"] = {2, 54};
    // cpu_id_map["0x1d0002000000000f"] = {2, 55};
    // cpu_id_map["0x1d00020000000010"] = {2, 56};
    // cpu_id_map["0x1d00020000000011"] = {2, 57};
    // cpu_id_map["0x1d00020000000012"] = {2, 58};
    // cpu_id_map["0x1d00020000000013"] = {2, 59};

    // cpu_id_map["0x1d00030000000000"] = {3, 60};
    // cpu_id_map["0x1d00030000000001"] = {3, 61};
    // cpu_id_map["0x1d00030000000002"] = {3, 62};
    // cpu_id_map["0x1d00030000000003"] = {3, 63};
    // cpu_id_map["0x1d00030000000004"] = {3, 64};
    // cpu_id_map["0x1d00030000000005"] = {3, 65};
    // cpu_id_map["0x1d00030000000006"] = {3, 66};
    // cpu_id_map["0x1d00030000000007"] = {3, 67};
    // cpu_id_map["0x1d00030000000008"] = {3, 68};
    // cpu_id_map["0x1d00030000000009"] = {3, 69};
    // cpu_id_map["0x1d0003000000000a"] = {3, 70};
    // cpu_id_map["0x1d0003000000000b"] = {3, 71};
    // cpu_id_map["0x1d0003000000000c"] = {3, 72};
    // cpu_id_map["0x1d0003000000000d"] = {3, 73};
    // cpu_id_map["0x1d0003000000000e"] = {3, 74};
    // cpu_id_map["0x1d0003000000000f"] = {3, 75};
    // cpu_id_map["0x1d00030000000010"] = {3, 76};
    // cpu_id_map["0x1d00030000000011"] = {3, 77};
    // cpu_id_map["0x1d00030000000012"] = {3, 78};
    // cpu_id_map["0x1d00030000000013"] = {3, 79};

    // // set up mem_id_map for system mem : socket_id
    // mem_id_map["0x1e00000000000000"] = {0};
    // mem_id_map["0x1e00010000000000"] = {1};
    // mem_id_map["0x1e00020000000000"] = {2};
    // mem_id_map["0x1e00030000000000"] = {3};

}
// create machine model (sherlock)
EnhancedMachineModel *create_enhanced_machine_model()
{
    EnhancedMachineModel *machine = new EnhancedMachineModel("/Users/xluo/Documents/simulator/final/machine_config");
    machine->default_seg_size = 4194304;
    machine->max_num_segs = 10;
    machine->realm_comm_overhead = 0.1;
    std::cout << machine->to_string() << std::endl;
    init_id_maps();
    return machine;
}

SimpleMachineModel *create_simple_machine_model()
{
    SimpleMachineModel *machine = new SimpleMachineModel(4, 20, 4);
    std::cout << machine->to_string() << std::endl;
    init_id_maps();
    return machine;
}

void run_dag_file(int argc, char **argv)
{
    EnhancedMachineModel *machine = create_enhanced_machine_model();
    // SimpleMachineModel *machine = create_simple_machine_model();
    Simulator simulator((MachineModel *) machine);

    string folder;
    if (argc == 2) {
        folder = argv[1];
    }
    unordered_map<int, float> cost_map;
    // get costs of tasks
    std::ifstream cost_file(folder + "/cost");
    int uid;
    float cost;
    while (cost_file >> uid >> cost) {
        // cout << uid << " " << cost << endl;
        if (cost_map.find(uid) == cost_map.end()) {
            cost_map[uid] = cost * 0.001;  //us -> ms
        }
        else {
            cout << "Has duplicate uid in cost file" << endl;
        }
    }

    int num_comp_tasks = 0;
    int num_comm_tasks = 0;
    unordered_map<string, Task *> comp_tasks_map;
    unordered_map<string, long> messages_map;
    unordered_set<string> left;
    unordered_set<string> right;
    unordered_map<int, int> alias_map; 
    // get alias
    std::ifstream alias_file(folder + "/alias");
    if (alias_file.is_open()) {
        std::string line;
        while (std::getline(alias_file, line)) {
            vector<string> line_array = split(line, " ");
            if (line_array[0] == "alias:") {
                int t0 = stoi(line_array[1]);
                int t1 = stoi(line_array[2]);
                alias_map[t0] = t1;
                alias_map[t1] = t0;
            }
        }
    }

    // get comp tasks
    std::ifstream comp_file(folder + "/comp");
    if (comp_file.is_open()) {
        std::string line;
        while (std::getline(comp_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "comp:") {
                num_comp_tasks++;
                int task_id = -1;
                bool is_main = false;
                bool is_skip = false;
                CompDevice *comp_device = nullptr;
                MemDevice *mem_device = nullptr;
                if ((line_array[1] == "Conv2D" and line_array[2] == "Forward")
                    // or (line_array[1] == "SGD" and line_array[2] == "Parameter")
                    ) {
                    is_main = true;
                }
                // if ((line_array[1] == "Conv2D" and line_array[2] == "Init")  
                //     or (line_array[1] == "Load" and line_array[2] == "Entire" and line_array[3] == "Dataset") 
                //     //or (line_array[1] == "Zero" and line_array[2] == "Init")
                //     ) {
                //     is_skip = true;
                // }
                for (int i = 0; i < line_array.size(); i++) {
                    // skip some init tasks
                    if (line_array[i] == "(UID:") {
                        task_id = stoi(line_array[i+1].substr(0, line_array[i+1].size() - 1));
                    }
                    if (line_array[i] == "Processor:") {
                        if (line_array[i+1] == "CPU") {
                            pair<int, int> ids = cpu_id_map[line_array.back()];
                            comp_device = machine->get_cpu(ids.second);
                            // TODO: set up mem from comm
                            mem_device = machine->get_sys_mem(ids.first);
                        }
                        else if (line_array[i+1] == "GPU") {
                            pair<int, int> ids = gpu_id_map[line_array.back()];
                            comp_device = machine->get_gpu(ids.second);
                            mem_device = machine->get_gpu_fb_mem(ids.second);
                        }
                        else {
                            cout << "Unknow type of processor" << endl;
                        }
                        break;
                    }
                }
                string task_name = "op_node_" + to_string(task_id);
                float cost = 0.0;
                if (cost_map.find(task_id) != cost_map.end()) {
                    cost = cost_map[task_id];
                }
                else {
                    cost = cost_map[alias_map[task_id]];
                    cout << "========= task " << task_id << " has alias " << alias_map[task_id] << " with cost " << cost << endl;
                }
                if (is_skip) {
                    cost = 0.0;
                }
                Task *cur_task = simulator.new_comp_task(task_name, comp_device, cost, mem_device);
                cur_task->is_main = is_main;
                // cout << cur_task->to_string() << endl;
                comp_tasks_map[task_name] = cur_task;
            }
            else {
                cout << "error" << endl;
            }
        }
        comp_file.close();
    }
    // get comm tasks
    std::ifstream comm_file(folder + "/comm");
    if (comm_file.is_open()) {
        std::string line;
        while (std::getline(comm_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "comm:") {
                num_comm_tasks++;
                int loc = 2;
                if (line_array[loc] == "'Realm" and (line_array[loc+1] == "Copy" or line_array[loc+1] == "Fill")) {
                    string task_name;
                    string comp_device_id = line_array.back().substr(0, line_array.back().size() - 3);
                    string comp_device_type = line_array[line_array.size()-3];
                    comp_device_type = comp_device_type.substr(comp_device_type.size()-3, 3);
                    string realm_id = line_array[loc+2].substr(1, line_array[loc+2].size() - 2);
                    string src_mem_device_id = "";
                    string src_mem_device_type = "";
                    string tar_mem_device_id = "";
                    string tar_mem_device_type = "";
                    // cout << realm_id << endl;
                    if (line_array[loc+1] == "Copy") {
                        task_name = "realm_copy_" + realm_id;
                        src_mem_device_id = line_array[line_array.size()-11];
                        src_mem_device_id = src_mem_device_id.substr(0, src_mem_device_id.size()-1);
                        src_mem_device_type = line_array[line_array.size()-13];
                        tar_mem_device_id = line_array[line_array.size()-4];
                        tar_mem_device_id = tar_mem_device_id.substr(0, tar_mem_device_id.size()-1);
                        tar_mem_device_type = line_array[line_array.size()-6];
                    }
                    else {
                        task_name = "realm_fill_" + realm_id;
                        src_mem_device_id = line_array[line_array.size()-4];
                        src_mem_device_id = src_mem_device_id.substr(0, src_mem_device_id.size()-1);
                        src_mem_device_type = line_array[line_array.size()-6];

                    }
                    // cout << task_name << " " << comp_device_type << "-" << comp_device_id << " " << src_mem_device_type << "-" << src_mem_device_id << " " << tar_mem_device_type << "-" << tar_mem_device_id << endl;
                    Task *cur_task;
                    CompDevice *comp_device;
                    MemDevice *mem_device;
                    int random_util_id = rand() % utils_ids[0].size();
                    if (comp_device_type == "CPU") {
                        pair<int, int> ids = cpu_id_map[comp_device_id];
                        // comp_device = machine->get_cpu(ids.second);
                        pair<int, int> temp_ids = cpu_id_map[utils_ids[ids.first][random_util_id]];
                        comp_device = machine->get_cpu(temp_ids.second);
                    }
                    else if (comp_device_type == "GPU") {
                        pair<int, int> ids = gpu_id_map[comp_device_id];
                        comp_device = machine->get_gpu(ids.second);
                    }
                    else {
                        cout << "wrong comp_device_type" << endl;
                        assert(0);
                    }
                    if (src_mem_device_type == "System" or src_mem_device_type == "Zero-Copy") {
                        int socket_id = mem_id_map[src_mem_device_id];
                        mem_device = machine->get_sys_mem(socket_id);
                    }
                    else if (src_mem_device_type == "Framebuffer") {
                        int device_id = mem_id_map[src_mem_device_id];
                        mem_device = machine->get_gpu_fb_mem(device_id);
                    }
                    else {
                        cout << "wrong src_mem_device_type" << endl;
                        assert(0);
                    }
                    cur_task = simulator.new_comp_task(task_name, comp_device, machine->realm_comm_overhead, mem_device);
                    // cout << cur_task->to_string() << endl;
                    comp_tasks_map[task_name] = cur_task;
                    long index_size = 0, field_size = 0;
                    string task_uid = "";
                    for (int i = 0; i < line_array.size(); i++) {
                        if (line_array[i] == "Index_Space_Size:") {
                            index_size = stol(line_array[i+1]);
                        }
                        if (line_array[i] == "Field_Size:") {
                            field_size = stol(line_array[i+1]);
                            break;
                        }
                        // Realm Fill do not have "UID:" in comm
                        if (line_array[i] == "(UID:") {
                            task_uid = line_array[i+1];
                            task_uid = task_uid.substr(0, task_uid.size()-3);
                        }
                    }
                    assert(index_size > 0 and field_size > 0);
                    messages_map[task_name] = index_size * field_size;
                    // cout << task_name << " - " << index_size * field_size << " bytes" << endl;
                    
                    // change op_node tasks' memory based on the tar_mem_device_id
                    // TODO: do not change op_node tasks' memory, use the target memory in the comm directly
                    string tar_task_name = "op_node_" + task_uid;
                    // cout << tar_task_name << endl;
                    if (comp_tasks_map.find(tar_task_name) != comp_tasks_map.end()) {
                        CompTask *tar_comp_task = (CompTask *)comp_tasks_map.at(tar_task_name);
                        MemDevice *tar_mem_device;
                        if (tar_mem_device_type == "System" or tar_mem_device_type == "Zero-Copy") {
                            int socket_id = mem_id_map[tar_mem_device_id];
                            tar_mem_device = machine->get_sys_mem(socket_id);
                        }
                        else if (tar_mem_device_type == "Framebuffer") {
                            int device_id = mem_id_map[tar_mem_device_id];
                            tar_mem_device = machine->get_gpu_fb_mem(device_id);
                        }
                        else {
                            cout << "wrong tar_mem_device_type" << endl;
                            assert(0);
                        }
                        tar_comp_task->mem = tar_mem_device;
                        // cout << tar_comp_task->to_string() << endl;
                    }

                }
                else {
                    cout << line << endl;
                    cout << "comm: has other types" << endl;
                    assert(0);
                }
            }
            else {
                cout << "error" << endl;
            }
        }
        comp_file.close();
    }    
    // get deps
    std::ifstream deps_file(folder + "/deps");
    if (deps_file.is_open()) {
        std::string line;
        while (std::getline(deps_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "deps:") {
                // cout << line << endl;
                //cout << line_array[1] << " " << line_array[3] << endl;
                if (comp_tasks_map.find(line_array[1]) != comp_tasks_map.end() and
                    comp_tasks_map.find(line_array[3]) != comp_tasks_map.end()) {
                    long message_size = 0;
                    if (starts_with(line_array[1], "realm")) {
                        message_size = messages_map[line_array[1]];
                    }
                    else if (starts_with(line_array[3], "realm")) {
                        message_size = 0;
                    }
                    simulator.new_comm_task(comp_tasks_map[line_array[1]], comp_tasks_map[line_array[3]], message_size);
                    if (left.find(line_array[1]) == left.end()) {
                        left.insert(line_array[1]);
                    }
                    if (right.find(line_array[3]) == right.end()) {
                        right.insert(line_array[3]);
                    }
                }
                else {
                    if (comp_tasks_map.find(line_array[1]) == comp_tasks_map.end())
                        // cout << "deps: cannot find " << line_array[1] << endl;
                        ;
                    if (comp_tasks_map.find(line_array[3]) == comp_tasks_map.end())
                        // cout << "deps: cannot find " << line_array[3] << endl;
                        ;
                }

            }
            else {
                cout << "error" << endl;
            }
        }
        deps_file.close();
    }

    for (auto i : left) {
        if (right.find(i) == right.end()) {
            cout << "starts with:" << i << endl;
            simulator.enter_ready_queue(comp_tasks_map[i]);
        }
    }
    
    simulator.simulate();
    cout << "num_comp_tasks " << num_comp_tasks << endl;
    cout << "num_comm_tasks " << num_comm_tasks << endl;

}

void test_comm()
{
    EnhancedMachineModel *machine = create_enhanced_machine_model();
    // SimpleMachineModel *machine = create_simple_machine_model();
    cout << machine->to_string();
    Simulator simulator((MachineModel *) machine);
    int num_nodes = 2;
    int num_sockets_per_node = 1;
    int num_cpus_per_socket = 20;
    int num_gpus_per_socket = 4;

    vector<Task *> comp_tasks;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_sockets_per_node; j++) {
            for (int k = 0; k < num_cpus_per_socket; k++) {
                int socket_id = i * num_sockets_per_node + j;
                int task_id = socket_id * num_cpus_per_socket + k;
                string task_name = "comp_task " + to_string(task_id) + " on socket " + to_string(socket_id) + " cpu " + to_string(k);
                float run_time = 0;
                comp_tasks.emplace_back(simulator.new_comp_task(task_name, machine->get_cpu(task_id), run_time, machine->get_sys_mem(socket_id)));
            }
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_sockets_per_node; j++) {
            for (int k = 0; k < num_gpus_per_socket; k++) {
                int socket_id = i * num_sockets_per_node + j;
                int task_id = socket_id * num_gpus_per_socket + k;
                string task_name = "comp_task " + to_string(task_id) + " on socket " + to_string(socket_id) + " gpu " + to_string(k);;
                float run_time = 0;
                comp_tasks.emplace_back(simulator.new_comp_task(task_name, machine->get_gpu(task_id), run_time, machine->get_gpu_fb_mem(task_id)));
            }
        }
    }

    Task *src = comp_tasks[0];
    Task *tar = comp_tasks[20];
    long message_size = 64 << 20;
    simulator.enter_ready_queue(src);
    simulator.new_comm_task(src, tar, message_size);
    simulator.simulate();
}

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    run_dag_file(argc, argv);
    // stencil_1d_cpu();
    // test_comm();
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
    cout << "simulator runs: " << time_span.count() << " seconds" <<  endl;
}
