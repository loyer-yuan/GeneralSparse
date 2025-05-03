#include "../operator.hpp"

thread_bit_map_operator::thread_bit_map_operator(shared_ptr<code_generator> code_generator_ptr, POS_TYPE pos, unsigned int size, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor, shared_ptr<operator_context> operator_history)
    : basic_operator("thread_bit_map_operator", code_generator_ptr->get_metadata_set(), IMPLEMENTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    assert(coarsen_factor <= 16);
    this->pos = pos;
    this->coarsen_factor = coarsen_factor;
    this->sparse_coarsen_factor = sparse_coarsen_factor;
    this->size = size;
    this->code_generator_ptr = code_generator_ptr;
    this->code_generator_ptr->open_spec_level_of_paral(THREAD_META);

}

bool thread_bit_map_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool thread_flag = false;
    bool nnz_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        for (int i = 0; i < former_operator_distributing.size(); i++)
        {
            if (former_operator_distributing[i]->get_name().find("nnz") != string::npos)
            {
                nnz_flag = true;
            }

            if (former_operator_distributing[i]->get_name().find("thread") != string::npos)
            {
                thread_flag = true;
            }
        }
    }
    if (thread_flag == true && nnz_flag == true)
    {
        return true;
    }

    return false;
}
bool thread_bit_map_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    if (this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", this->target_matrix_id) == false)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void thread_bit_map_operator::run(bool check)
{
    assert(this->is_valid_according_to_metadata() == true);

    
    bool parent_flag = false;
    if(this->pos != THREAD_META)
    {
        parent_flag = true;
    }

    shared_ptr<thread_bit_map> thread_bit_map_ptr(new thread_bit_map(this->meta_data_set_ptr, parent_flag, this->size, this->target_matrix_id));
    thread_bit_map_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(thread_bit_map_ptr));

    shared_ptr<segment_empty_flag> segment_empty_flag_ptr(new segment_empty_flag(this->meta_data_set_ptr, this->pos, this->size, this->target_matrix_id));
    segment_empty_flag_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(segment_empty_flag_ptr));

    shared_ptr<segment_empty_row_indices> segment_empty_row_indices_ptr(new segment_empty_row_indices(this->meta_data_set_ptr, this->pos, this->target_matrix_id));
    segment_empty_row_indices_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(segment_empty_row_indices_ptr));

    shared_ptr<segment_offset> segment_offset_ptr(new segment_offset(this->meta_data_set_ptr, false, this->size, this->target_matrix_id));
    segment_offset_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(segment_offset_ptr));

    shared_ptr<segment_ptr> segment_ptr_ptr(new segment_ptr(this->meta_data_set_ptr, this->pos, this->target_matrix_id));
    segment_ptr_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(segment_ptr_ptr));

    bool need_warp_reduction = false;
    if(this->pos == WARP_META)
    {
        need_warp_reduction = true;
    }
    shared_ptr<thread_bit_map_reduce_to_two_register_token> thread_bit_map_token_ptr(new thread_bit_map_reduce_to_two_register_token(this->meta_data_set_ptr, need_warp_reduction, this->sparse_coarsen_factor, this->coarsen_factor, this->code_generator_ptr));
    this->code_generator_ptr->set_reduction_token(THREAD_META, thread_bit_map_token_ptr);
    this->code_generator_ptr->open_spec_level_of_paral(THREAD_META);

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> thread_bit_map_operator::get_data_transform_sequence()
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

string thread_bit_map_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "thread_bit_map_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}