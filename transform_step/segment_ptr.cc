#include "data_transform_step.hpp"

segment_ptr::segment_ptr(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int target_matrix_id)
    : basic_data_transform_step("segment_ptr", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
    this->pos = pos;
}

void segment_ptr::run(bool check)
{
     if (check)
    {
        assert(this->target_matrix_id >= 0);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices", this->target_matrix_id));

        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", this->target_matrix_id));
    }

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> first_nz_indices_record(new data_item_record(THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_record);


    // 当前的行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    shared_ptr<universal_array> first_nzs_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();


    vector<unsigned long> segment_ptr_vec;
    segment_ptr_vec.push_back(0);
    unsigned long count = 0;

    for (unsigned long j = 0; j < first_nzs_ptr->get_len() - 2; j++)
    {
        unsigned long cur_nz = first_nzs_ptr->read_integer_from_arr(j);
        unsigned long next_nz = first_nzs_ptr->read_integer_from_arr(j + 1);
        unsigned long cur_first_row = nz_row_indices_ptr->read_integer_from_arr(cur_nz);
        count += 1;
        for (unsigned int i = cur_nz + 1; i < next_nz; i++)
        {
            unsigned long cur_row = nz_row_indices_ptr->read_integer_from_arr(i);
            unsigned long former_row = nz_row_indices_ptr->read_integer_from_arr(i - 1);
            if (cur_row != former_row)
            {
                count += 1;
            }
        }
        segment_ptr_vec.push_back(count);
    }
    assert(segment_ptr_vec.size() == first_nzs_ptr->get_len() - 1);

    // 将数据放到metadata set中
    shared_ptr<universal_array> segment_ptr_ptr(new universal_array(&(segment_ptr_vec[0]), segment_ptr_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> segment_ptr_item(new meta_data_item(segment_ptr_ptr, THREAD_META, "segment_ptr", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(segment_ptr_item);

    // 执行记录
    shared_ptr<data_item_record> segment_ptr_record(new data_item_record(THREAD_META, "segment_ptr", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(segment_ptr_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> segment_ptr::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> segment_ptr::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string segment_ptr::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "segment_ptr::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + "}";

    return return_str;
}