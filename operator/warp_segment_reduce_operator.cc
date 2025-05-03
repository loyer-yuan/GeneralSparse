#include "../operator.hpp"

warp_segment_reduce_operator::warp_segment_reduce_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor, bool relative_nz, bool relative_row, shared_ptr<operator_context> operator_history)
    : basic_operator("warp_segment_reduce_operator", code_generator_ptr->get_metadata_set(), IMPLEMENTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    assert(coarsen_factor <= 16);
    this->coarsen_factor = coarsen_factor;
    this->code_generator_ptr = code_generator_ptr;
    this->code_generator_ptr->open_spec_level_of_paral(WARP_META);
    this->relative_nz = relative_nz;
    this->relative_row = relative_row;
}

bool warp_segment_reduce_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool thread_flag = false;
    bool nnz_flag = false;
    bool warp_flag = false;
    bool thread_bit_map_flag = false;

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

        if (former_operator_distributing[i]->get_name().find("warp") != string::npos)
        {
            warp_flag = true;
        }

    }

    for (int i = 0; i < former_operator_implementing.size(); i++)
    {
        if (former_operator_implementing[i]->get_name().find("thread_bit_map") != string::npos)
        {
            thread_bit_map_flag = true;
        }
    }

    if (thread_flag == true && nnz_flag == true && warp_flag == false && thread_bit_map_flag == true)
    {
        return true;
    }

    return false;
}
bool warp_segment_reduce_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    if (this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", this->target_matrix_id) == false)
    {
        return false;
    }
    else if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id) == true)
    {
        return false;
    }
    return true;
}

void warp_segment_reduce_operator::run(bool check)
{
    assert(this->is_valid_according_to_metadata() == true);

    shared_ptr<get_begin_rows_after_merge_thread> get_begin_rows_after_merge_thread_ptr(new get_begin_rows_after_merge_thread(this->meta_data_set_ptr, WARP_META, get_config()["VECTOR_WIDTH"].as_float(), this->target_matrix_id));
    get_begin_rows_after_merge_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_rows_after_merge_thread_ptr));

    shared_ptr<get_begin_nzs_after_merge_thread> get_begin_nzs_after_merge_thread_ptr(new get_begin_nzs_after_merge_thread(this->meta_data_set_ptr, WARP_META, get_config()["VECTOR_WIDTH"].as_float(), this->target_matrix_id));
    get_begin_nzs_after_merge_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_nzs_after_merge_thread_ptr));

    if (this->relative_row == true)
    {
        shared_ptr<get_begin_rows_relative_to_parent_after_merge_thread> get_begin_rows_relative_to_parent_after_merge_thread_ptr(new get_begin_rows_relative_to_parent_after_merge_thread(this->meta_data_set_ptr, WARP_META, get_config()["VECTOR_WIDTH"].as_float(), this->target_matrix_id));
        get_begin_rows_relative_to_parent_after_merge_thread_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_rows_relative_to_parent_after_merge_thread_ptr));
    }

    if (this->relative_nz == true)
    {
        shared_ptr<get_begin_nzs_relative_to_parent_after_merge_thread> get_begin_nzs_relative_to_parent_after_merge_thread_ptr(new get_begin_nzs_relative_to_parent_after_merge_thread(this->meta_data_set_ptr, WARP_META, get_config()["VECTOR_WIDTH"].as_float(), this->target_matrix_id));
        get_begin_nzs_relative_to_parent_after_merge_thread_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_nzs_relative_to_parent_after_merge_thread_ptr));
    }


    shared_ptr<get_begin_BMTs_after_merge_thread> get_begin_BMTs_after_merge_thread_ptr(new get_begin_BMTs_after_merge_thread(this->meta_data_set_ptr, WARP_META, get_config()["VECTOR_WIDTH"].as_float(), this->target_matrix_id));
    get_begin_BMTs_after_merge_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMTs_after_merge_thread_ptr));

    shared_ptr<warp_segment_reduce_token> warp_segment_reduce_ptr(new warp_segment_reduce_token(this->meta_data_set_ptr, this->coarsen_factor, this->code_generator_ptr));

    this->code_generator_ptr->set_reduction_token(WARP_META, warp_segment_reduce_ptr);
    this->code_generator_ptr->open_spec_level_of_paral(WARP_META);

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> warp_segment_reduce_operator::get_data_transform_sequence()
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

string warp_segment_reduce_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "warp_segment_reduce_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}