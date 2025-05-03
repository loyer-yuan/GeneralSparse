#include "../operator.hpp"

tblock_total_reduce_operator::tblock_total_reduce_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor, shared_ptr<operator_context> operator_history)
    : basic_operator("tblock_total_reduce_operator", code_generator_ptr->get_metadata_set(), IMPLEMENTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    assert(coarsen_factor <= 16);
    this->coarsen_factor = coarsen_factor;
    this->code_generator_ptr = code_generator_ptr;
    this->code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);

}

bool tblock_total_reduce_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool tblock_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        for (int i = 0; i < former_operator_distributing.size(); i++)
        {
            if (former_operator_distributing[i]->get_name().find("nnz") != string::npos)
            {
                return false;
            }

            if (former_operator_distributing[i]->get_name().find("tblock") != string::npos)
            {
                tblock_flag = true;
            }
        }
    }
    if (tblock_flag == true)
    {
        return true;
    }

    return false;
}
bool tblock_total_reduce_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    if (this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id) == false)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void tblock_total_reduce_operator::run(bool check)
{
    assert(this->is_valid_according_to_metadata() == true);

    shared_ptr<total_block_result_reduce_to_one_register_token> total_BMTB_reduce(new total_block_result_reduce_to_one_register_token(this->meta_data_set_ptr, this->coarsen_factor, this->code_generator_ptr));

    this->code_generator_ptr->set_reduction_token(TBLOCK_META, total_BMTB_reduce);
    this->code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> tblock_total_reduce_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    // 检查所有的内容
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回对应的操作序列
    return this->transform_seq;
}

string tblock_total_reduce_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "tblock_total_reduce_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}