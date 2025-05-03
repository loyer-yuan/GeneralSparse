#include "data_transform_graph.hpp"

// 往transform_graph中是增加新的新的数据转化的工作
void transform_graph::add_transform_step(shared_ptr<transform_step_record_item> item)
{
    assert(item != NULL);

    // 首先查看最大的step，step从1开始
    int cur_max_step = 0;

    map<int, shared_ptr<transform_step_record_item>>::iterator iter;
    iter = this->transform_step_item_in_each_step.begin();

    while (iter != this->transform_step_item_in_each_step.end())
    {
        if (cur_max_step < iter->first)
        {
            cur_max_step = iter->first;
        }
        iter++;
    }

    // 将最大步骤自增1来获得最新的步骤
    int new_step = cur_max_step + 1;

    // new_step作为key在当前map中不存在
    assert(this->transform_step_item_in_each_step.count(new_step) == 0);
    this->transform_step_item_in_each_step[new_step] = item;
}

void transform_graph::add_transform_step(vector<shared_ptr<transform_step_record_item>> item_vec)
{
    // 数组中要有内容
    assert(item_vec.size() > 0);

    // 遍历数组中所有元素
    for (unsigned long i = 0; i < item_vec.size(); i++)
    {
        this->add_transform_step(item_vec[i]);
    }
}

shared_ptr<transform_step_record_item> transform_graph::find_transform_step(int step)
{
    // 不存在会导致错误
    assert(this->transform_step_item_in_each_step.count(step) != 0);
    // 将内容读出
    return this->transform_step_item_in_each_step[step];
}

void transform_graph::remove_transform_step(int step)
{
    // 去除一些数据转换的步骤
    assert(this->transform_step_item_in_each_step.count(step) != 0);
    // 删除元素
    this->transform_step_item_in_each_step.erase(step);
}

int transform_graph::get_max_step()
{
    int cur_max_step = 0;
    map<int, shared_ptr<transform_step_record_item>>::iterator iter;
    iter = this->transform_step_item_in_each_step.begin();

    while (iter != this->transform_step_item_in_each_step.end())
    {
        if (cur_max_step < iter->first)
        {
            cur_max_step = iter->first;
        }
        iter++;
    }

    return cur_max_step;
}