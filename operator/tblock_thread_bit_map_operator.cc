#include "../operator.hpp"

tblock_thread_bit_map_operator::tblock_thread_bit_map_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor, int block_size, bool relative_nz, bool relative_row, shared_ptr<operator_context> operator_history)
    : basic_operator("tblock_thread_bit_map_operator", code_generator_ptr->get_metadata_set(), IMPLEMENTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    assert(coarsen_factor <= 16);
    this->coarsen_factor = coarsen_factor;
    this->code_generator_ptr = code_generator_ptr;
    this->code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);
    this->block_size = block_size;
    this->relative_nz = relative_nz;
    this->relative_row = relative_row;
}

bool tblock_thread_bit_map_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool thread_flag = false;
    bool tblock_flag = false;

    for (int i = 0; i < former_operator_distributing.size(); i++)
    {
        if (former_operator_distributing[i]->get_name().find("nnz") != string::npos)
        {
            return false;
        }

        if (former_operator_distributing[i]->get_name().find("thread") != string::npos)
        {
            thread_flag = true;
        }

        if (former_operator_distributing[i]->get_name().find("tblock") != string::npos)
        {
            tblock_flag = true;
        }
    }

    if (thread_flag == true && tblock_flag == false)
    {
        return true;
    }

    return false;
}
bool tblock_thread_bit_map_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    if (this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", this->target_matrix_id) == false)
    {
        return false;
    }

    return true;
}

void tblock_thread_bit_map_operator::run(bool check)
{
    assert(this->is_valid_according_to_metadata() == true);

    shared_ptr<get_begin_rows_after_merge_thread> get_begin_rows_after_merge_thread_ptr(new get_begin_rows_after_merge_thread(this->meta_data_set_ptr, TBLOCK_META, this->block_size, this->target_matrix_id));
    get_begin_rows_after_merge_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_rows_after_merge_thread_ptr));


    shared_ptr<get_begin_nzs_after_merge_thread> get_begin_nzs_after_merge_thread_ptr(new get_begin_nzs_after_merge_thread(this->meta_data_set_ptr, TBLOCK_META, this->block_size, this->target_matrix_id));
    get_begin_nzs_after_merge_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_nzs_after_merge_thread_ptr));

    if (this->relative_row)
    {
        shared_ptr<get_begin_rows_relative_to_parent_after_merge_thread> get_begin_rows_relative_to_parent_after_merge_thread_ptr(new get_begin_rows_relative_to_parent_after_merge_thread(this->meta_data_set_ptr, TBLOCK_META, this->block_size, this->target_matrix_id));
        get_begin_rows_relative_to_parent_after_merge_thread_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_rows_relative_to_parent_after_merge_thread_ptr));
    }

    if (relative_nz)
    {
        shared_ptr<get_begin_nzs_relative_to_parent_after_merge_thread> get_begin_nzs_relative_to_parent_after_merge_thread_ptr(new get_begin_nzs_relative_to_parent_after_merge_thread(this->meta_data_set_ptr, TBLOCK_META, this->block_size, this->target_matrix_id));
        get_begin_nzs_relative_to_parent_after_merge_thread_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_nzs_relative_to_parent_after_merge_thread_ptr));
    }

    shared_ptr<get_begin_BMTs_after_merge_thread> get_begin_BMTs_after_merge_thread_ptr(new get_begin_BMTs_after_merge_thread(this->meta_data_set_ptr, TBLOCK_META, this->block_size, this->target_matrix_id));
    get_begin_BMTs_after_merge_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMTs_after_merge_thread_ptr));



    shared_ptr<parent_bit_map_of_thread> parent_bit_map_of_thread_ptr(new parent_bit_map_of_thread(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
    parent_bit_map_of_thread_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(parent_bit_map_of_thread_ptr));

    shared_ptr<segment_offset> segment_offset_ptr(new segment_offset(this->meta_data_set_ptr, TBLOCK_META, this->block_size, this->target_matrix_id));
    segment_offset_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(segment_offset_ptr));

    shared_ptr<tblock_bit_map_reduce_token> tblock_bit_map_reduce_ptr(new tblock_bit_map_reduce_token(this->meta_data_set_ptr, this->coarsen_factor, this->code_generator_ptr));

    this->code_generator_ptr->set_reduction_token(TBLOCK_META, tblock_bit_map_reduce_ptr);
    this->code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);
    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> tblock_thread_bit_map_operator::get_data_transform_sequence()
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

string tblock_thread_bit_map_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "tblock_thread_bit_map_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}