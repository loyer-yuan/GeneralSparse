#include "operator.hpp"

bool check_operator_stage_type(operator_stage_type type)
{
    if (type == CONVERTING_OP)
    {
        return true;
    }

    if (type == DISTRIBUTING_OP)
    {
        return true;
    }

    if (type == IMPLEMENTING_OP)
    {
        return true;
    }

    if (type == NONE_OP)
    {
        return true;
    }

    return false;
}

string convert_operator_stage_type_to_string(operator_stage_type type)
{
    assert(check_operator_stage_type(type));

    if (type == CONVERTING_OP)
    {
        return "CONVERTING_OP";
    }

    if (type == DISTRIBUTING_OP)
    {
        return "DISTRIBUTING_OP";
    }

    if (type == IMPLEMENTING_OP)
    {
        return "IMPLEMENTING_OP";
    }

    if (type == NONE_OP)
    {
        return "NONE_OP";
    }

    assert(false);
    return "";
}

// transform_step_item：transform_step_item
transform_step_record_item::transform_step_record_item(vector<shared_ptr<data_item_record>> source_data_item_ptr_vec, vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec, shared_ptr<basic_data_transform_step> transform_step_ptr)
{
    // 输入数据中不能有空指针
    for (unsigned long i = 0; i < source_data_item_ptr_vec.size(); i++)
    {
        assert(source_data_item_ptr_vec[i] != NULL);
    }

    // 输出数据没有空指针
    for (unsigned long i = 0; i < dest_data_item_ptr_vec.size(); i++)
    {
        assert(dest_data_item_ptr_vec[i] != NULL);
    }
    // 边
    assert(transform_step_ptr != NULL);

    this->source_data_item_ptr_vec = source_data_item_ptr_vec;
    this->dest_data_item_ptr_vec = dest_data_item_ptr_vec;
    this->transform_step_ptr = transform_step_ptr;
}

string transform_step_record_item::convert_to_string()
{
    // 返回的字符串
    string return_str = "";

    // 遍历所有的输入记录
    for (int i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        // 当前不可能是空指针
        assert(this->source_data_item_ptr_vec[i] != NULL);
        
        return_str = return_str + "input" + to_string(i) + "-->" + this->source_data_item_ptr_vec[i]->convert_to_string() + "\n";
    }

    // 遍历所有的输出记录
    for (int i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        // 当前不可能是空指针
        assert(this->dest_data_item_ptr_vec[i] != NULL);
        
        return_str = return_str + "output" + to_string(i) + "-->" + this->dest_data_item_ptr_vec[i]->convert_to_string() + "\n";
    }

    // 数据转换
    return_str = return_str + "data_transform-->" + this->transform_step_ptr->convert_to_string();

    // 返回字符串
    return return_str;
}

shared_ptr<transform_step_record_item> get_record_item_of_a_transform_step(shared_ptr<basic_data_transform_step> transform_step_ptr)
{
    // 查看输入的data transform step是不是合法的
    assert(transform_step_ptr != NULL);
    assert(transform_step_ptr->check());

    // 首先获取transform_step的输入记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    source_data_item_ptr_vec = transform_step_ptr->get_source_data_item_ptr_in_data_transform_step();

    // 然后获取输出数组
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;
    dest_data_item_ptr_vec = transform_step_ptr->get_dest_data_item_ptr_in_data_transform_step();

    // 创建一个数据转换step的记录
    shared_ptr<transform_step_record_item> return_ptr(new transform_step_record_item(source_data_item_ptr_vec, dest_data_item_ptr_vec, transform_step_ptr));
    return return_ptr;
}