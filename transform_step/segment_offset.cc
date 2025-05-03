#include "data_transform_step.hpp"

segment_offset::segment_offset(shared_ptr<meta_data_set> meta_data_set_ptr, bool parent_flag, int size, int target_matrix_id)
    : basic_data_transform_step("segment_offset", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    this->size = size;
    this->target_matrix_id = target_matrix_id;
    this->parent_flag = parent_flag;
}

void segment_offset::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);

        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "thread_bit_map", this->target_matrix_id) || this->meta_data_set_ptr->is_exist(THREAD_META, "bit_map_of_thread", this->target_matrix_id));
    }

    shared_ptr<universal_array> thread_bit_map_ptr = NULL;

    if (this->meta_data_set_ptr->is_exist(THREAD_META, "thread_bit_map", this->target_matrix_id))
    {
        shared_ptr<data_item_record> thread_bit_map_record(new data_item_record(THREAD_META, "thread_bit_map", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(thread_bit_map_record);
        thread_bit_map_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "thread_bit_map", this->target_matrix_id)->get_metadata_arr();
    }
    else if (this->meta_data_set_ptr->is_exist(THREAD_META, "bit_map_of_thread", this->target_matrix_id))
    {
        shared_ptr<data_item_record> thread_bit_map_record(new data_item_record(THREAD_META, "bit_map_of_thread", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(thread_bit_map_record);
        thread_bit_map_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "bit_map_of_thread", this->target_matrix_id)->get_metadata_arr();
    }



    vector<unsigned long> segment_offset_vec(thread_bit_map_ptr->get_len(), 0);
    unsigned long count = 0;
    unsigned long prev_true_id = 0;
    for (unsigned long j = 1; j < thread_bit_map_ptr->get_len(); j++)
    {
        unsigned long cur_bit_map = thread_bit_map_ptr->read_integer_from_arr(j);
        if(cur_bit_map == 0 && ((j % this->size != 0) || parent_flag == false))
        {
            count += 1;
        }
        else
        {
            segment_offset_vec[prev_true_id] = count;
            count = 0;
            prev_true_id = j;
        }
    }


    shared_ptr<universal_array> segment_offset_ptr(new universal_array(&(segment_offset_vec[0]), segment_offset_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> segment_offset_item(new meta_data_item(segment_offset_ptr, THREAD_META, "segment_offset", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(segment_offset_item);

    shared_ptr<data_item_record> segment_offset_record(new data_item_record(THREAD_META, "segment_offset", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(segment_offset_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> segment_offset::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> segment_offset::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string segment_offset::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "segment_offset::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + "}";

    return return_str;
}