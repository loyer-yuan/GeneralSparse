#ifndef DATA_TRANSFORM_GRAPH_HPP
#define DATA_TRANSFORM_GRAPH_HPP

#include "metadata_set.hpp"
#include "data_transform_step.hpp"
#include "operator.hpp"
#include <map>

using namespace std;

// 需要记录整个operator graph在metadata set中的实际执行过程，是一个带有时序，带有一定约束的计算图
// data transform graph本质上也是一张表，包含了每一个data transform的操作，记录了操作的输入数据名和输出数据名，操作的时序

// 创建一个表格来存储所有的所有的操作条目，key是int类型的步骤，value是一次transform_step
// 使用一张表格，来记录所有执行过的data transform
class transform_graph
{
public:
    // 加入元素
    void add_transform_step(shared_ptr<transform_step_record_item> item);

    // 加入多个data transform step的执行记录
    void add_transform_step(vector<shared_ptr<transform_step_record_item>> item_vec);
    
    // 查到对应步骤的元素，如果不存在会直接导致错误
    shared_ptr<transform_step_record_item> find_transform_step(int step);

    // 删除元素，如果不存在会直接导致错误
    void remove_transform_step(int step);

    // 查看对应步骤的元素是否存在
    bool is_exist(int step);

    // 获得最大的步骤
    int get_max_step();

    // 通过一个data transform step在数据流转图中加入一个数据变换的步骤
    // void add_transform_step(shared_ptr<basic_data_transform_step> data_transform_step_ptr);

private:
    map<int, shared_ptr<transform_step_record_item>> transform_step_item_in_each_step;
};

// 执行器，包含了operator graph的记录，执行的依赖，metadataset，data_transform_graph


#endif