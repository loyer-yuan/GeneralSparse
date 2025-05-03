#include "../operator.hpp"

calculation_method_choose_operator::calculation_method_choose_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int method)
    : basic_operator("calculation_method_choose_operator", meta_data_set_ptr, CHOOSING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(method >= 0 && method < 4);
    this->method = method;
}

calculation_method_choose_operator::calculation_method_choose_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int method, shared_ptr<operator_context> operator_history)
    : basic_operator("calculation_method_choose_operator", meta_data_set_ptr, CHOOSING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(method >= 0 && method < 4);
    this->method = method;


}

calculation_method_choose_operator::calculation_method_choose_operator(shared_ptr<code_generator> code_generator_ptr, int method, shared_ptr<operator_context> operator_history)
    : basic_operator("calculation_method_choose_operator", code_generator_ptr->get_metadata_set(), CHOOSING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this) calculation_method_choose_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), method, operator_history);
}

bool calculation_method_choose_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);

    bool self_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        if (former_operator_distributing.size() == 0)
        {
            return true;
        }
    }
    return false;
}

bool calculation_method_choose_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());

    return true;
}

// 执行具体的排序操作
void calculation_method_choose_operator::run(bool check)
{
    if (check == true)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->method >= 0 && this->method < 4);
    }
    
    vector<int> methods;
    methods.push_back(this->method);
    shared_ptr<universal_array> calculation(new universal_array(((void *)(&methods[0])), methods.size(), UNSIGNED_INT));
    shared_ptr<meta_data_item> calculation_method(new meta_data_item(calculation, GLOBAL_META, "calculation_method", target_matrix_id));
    this->meta_data_set_ptr->add_element(calculation_method);
    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> calculation_method_choose_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回所有的操作集合
    return this->transform_seq;
}

string calculation_method_choose_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "calculation_method_choose_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}