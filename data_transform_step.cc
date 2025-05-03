#include "data_transform_step.hpp"

// data_item_record：data_transform_graph中的节点，代表了在数据变换过程中数据项的描述
data_item_record::data_item_record(POS_TYPE meta_position, string name, int sub_matrix_id)
{
    // 类型检查
    assert(check_pos_type(meta_position));
    assert(name != "");
    assert(sub_matrix_id >= -1);
    assert(meta_position != NONE_META);

    // 赋值
    this->meta_position = meta_position;
    this->name = name;
    this->sub_matrix_id = sub_matrix_id;
}

string data_item_record::convert_to_string()
{
    // 对于当前的检查
    assert(check_pos_type(meta_position));

    // 写成一个map的形式，key为数据成员类型，value为数据成员内容
    string return_str = "data_item_record::{meta_position:" + convert_pos_type_to_string(this->meta_position) + ",";
    return_str = return_str + "name:\"" + name + "\",";
    return_str = return_str + "sub_matrix_id:" + to_string(sub_matrix_id) + "}";

    return return_str;
}