#include "data_transform_step.hpp"

segment_empty_flag::segment_empty_flag(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, unsigned int size, int target_matrix_id)
    : basic_data_transform_step("segment_empty_flag", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    this->size = size;
    this->target_matrix_id = target_matrix_id;
    this->pos = pos;
}

void segment_empty_flag::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", this->target_matrix_id));

    }

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> first_nz_indices_record(new data_item_record(THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_record);


    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> first_nz_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();


    vector<bool> flag;
    bool cur_over_flag = false;
    for (unsigned long j = 0; j < first_nz_ptr->get_len() - 1; j++)
    {
        // 当前BMW的行偏移
        unsigned long cur_nz = first_nz_ptr->read_integer_from_arr(j);
        unsigned long next_nz = first_nz_ptr->read_integer_from_arr(j + 1);
        cur_over_flag = false;
        for (unsigned int i = cur_nz + 1; i < next_nz; i++)
        {
            unsigned long cur_row = nz_row_indices_ptr->read_integer_from_arr(i);
            unsigned long former_row = nz_row_indices_ptr->read_integer_from_arr(i - 1);
            if (cur_row - former_row > 1)
            {
                flag.push_back(true);
                cur_over_flag = true;
                break;
            }
        }
        if (cur_over_flag == false)
        {
            flag.push_back(false);
        }
    }


    vector<unsigned long> segment_empty_flag_vec;
    for (unsigned long i = 0; i < flag.size(); i += this->size)
    {
        unsigned long cur_flag = 0;
        unsigned long k = (i + this->size - 1) > (flag.size() - 1) ? (flag.size() - 1) : (i + this->size - 1);
        for (unsigned int j = k; j >= i; j--)
        {

            cur_flag = (cur_flag << 1) | flag[j];
            if (j == 0)
            {
                break;
            }
        }
        segment_empty_flag_vec.push_back(cur_flag);
    }

    assert(flag.size() == first_nz_ptr->get_len() - 1);

    shared_ptr<universal_array> segment_empty_flag_ptr(new universal_array(&(segment_empty_flag_vec[0]), segment_empty_flag_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> segment_empty_flag_item(new meta_data_item(segment_empty_flag_ptr, THREAD_META, "segment_empty_flag", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(segment_empty_flag_item);

    shared_ptr<data_item_record> segment_empty_flag_record(new data_item_record(THREAD_META, "segment_empty_flag", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(segment_empty_flag_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> segment_empty_flag::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> segment_empty_flag::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string segment_empty_flag::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "segment_empty_flag::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + "}";

    return return_str;
}