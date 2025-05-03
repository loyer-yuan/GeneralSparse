#include "data_transform_step.hpp"

get_begin_rows_after_merge_thread::get_begin_rows_after_merge_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int merge_num, int target_matrix_id)
    : basic_data_transform_step("get_begin_rows_after_merge_thread", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
    this->pos = pos;
    this->merge_num = merge_num;
}

void get_begin_rows_after_merge_thread::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);

        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices", this->target_matrix_id) || this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices_without_ending", this->target_matrix_id));
    }

    shared_ptr<data_item_record> first_row_indices_record(new data_item_record(THREAD_META, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_row_indices_record);

    shared_ptr<universal_array> BMT_first_row_indices_ptr;
    if (this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices", this->target_matrix_id) == true)
    {
        BMT_first_row_indices_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
    }
    else
    {
        BMT_first_row_indices_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_row_indices_without_ending", this->target_matrix_id)->get_metadata_arr();
    }

    vector<unsigned long> parent_first_row_vec;

    // 遍历所有BMW
    for (unsigned long j = 0; j < BMT_first_row_indices_ptr->get_len() - 1; j+=this->merge_num)
    {
        unsigned long cur_row = BMT_first_row_indices_ptr->read_integer_from_arr(j);
        parent_first_row_vec.push_back(cur_row);
    }
    parent_first_row_vec.push_back(BMT_first_row_indices_ptr->read_integer_from_arr(BMT_first_row_indices_ptr->get_len() - 1));

    // 将数据放到metadata set中
    shared_ptr<universal_array> parent_first_row_ptr(new universal_array(&(parent_first_row_vec[0]), parent_first_row_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> parent_first_row_item(new meta_data_item(parent_first_row_ptr, this->pos, "first_row_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(parent_first_row_item);

    // 执行记录
    shared_ptr<data_item_record> parent_first_row_record(new data_item_record(this->pos, "first_row_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(parent_first_row_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_rows_after_merge_thread::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    // 空指针检查
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_rows_after_merge_thread::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_rows_after_merge_thread::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "get_begin_rows_after_merge_thread::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}