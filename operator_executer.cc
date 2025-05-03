#include "operator_executer.hpp"

operator_context::operator_context()
{
    
}

operator_executer::operator_executer()
{
    shared_ptr<operator_context> history_operator(new operator_context());
    this->operator_history = history_operator;
}

bool operator_executer::is_valid_of_inserted_operator(shared_ptr<basic_operator> current_operator)
{
    return current_operator->is_valid_according_to_operator(this->operator_history);
}

void operator_executer::add_and_run(shared_ptr<basic_operator> current_operator)
{
    assert(this->is_valid_of_inserted_operator(current_operator));
    current_operator->run();
    add_operator_context(current_operator);
    vector<shared_ptr<transform_step_record_item>> current_operator_transform_step = current_operator->get_data_transform_sequence();
    push_back_tranform_step_in_history(current_operator_transform_step);
}

void operator_executer::add_code_generator_to_specific_sub_matrix(int sub_matrix_id)
{
    // 当前必然存在一个元数据库
    assert(this->meta_data_set_ptr != NULL);

    // 查看当前的sub_matrix是不是真的存在，主要查看对应子矩阵的行列值是不是存在
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", sub_matrix_id));
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", sub_matrix_id));
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", sub_matrix_id));

    // 当前set不存在对应的代码生成器
    assert(this->all_code_generator.count(sub_matrix_id) == 0);

    // 初始化一个代码生成器
    shared_ptr<code_generator> new_code_generator_ptr(new code_generator(this->meta_data_set_ptr, sub_matrix_id));

    // 向set中加入对应的代码生成器
    this->all_code_generator[sub_matrix_id] = new_code_generator_ptr;
}

bool operator_executer::if_code_generator_exist(int sub_matrix_id)
{
    assert(sub_matrix_id >= 0);
    if (this->all_code_generator.count(sub_matrix_id) == 1)
    {
        return true;
    }

    return false;
}

shared_ptr<code_generator> operator_executer::get_code_generator_of_spec_sub_matrix(int sub_matrix_id)
{
    // 当前代码生成器必须存在
    assert(this->if_code_generator_exist(sub_matrix_id));

    return this->all_code_generator[sub_matrix_id];
}