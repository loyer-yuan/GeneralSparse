#include "data_transform_step.hpp"

thread_bit_map::thread_bit_map(shared_ptr<meta_data_set> meta_data_set_ptr, bool parent_flag, int parent_size, int target_matrix_id)
    : basic_data_transform_step("thread_bit_map", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
    this->parent_flag = parent_flag;
    this->parent_size = parent_size;
}

void thread_bit_map::run(bool check)
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


    shared_ptr<universal_array> BMT_first_nz_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();


    vector<bool> bit_map;
    bit_map.push_back(true);
    for (unsigned long j = 1; j < nz_row_indices_ptr->get_len(); j++)
    {
        unsigned long cur_row = nz_row_indices_ptr->read_integer_from_arr(j);
        unsigned long former_row = nz_row_indices_ptr->read_integer_from_arr(j - 1);

        if (cur_row != former_row)
        {
            bit_map.push_back(true);
        }
        else
        {
            bit_map.push_back(false);
        }
    }

    assert(bit_map.size() == nz_row_indices_ptr->get_len());
    assert(bit_map.size() == BMT_first_nz_ptr->read_integer_from_arr(BMT_first_nz_ptr->get_len() - 1));

    if (this->parent_flag == true)
    {
        unsigned long thread_size = 0;
        if(this->meta_data_set_ptr->is_exist(GLOBAL_META, "BMT_size_of_each_blk", this->target_matrix_id) == true)
        {
            thread_size = BMT_first_nz_ptr->read_integer_from_arr(1) - BMT_first_nz_ptr->read_integer_from_arr(0);
        }

        for (unsigned long i = 0; i < bit_map.size() / thread_size + 1; i += this->parent_size)
        {
            bit_map[i * thread_size] = true;
        }
    }

    vector<unsigned long> thread_bit_map_vec;
    for (unsigned long i = 0; i < BMT_first_nz_ptr->get_len() - 1; i++)
    {
        unsigned long map = 0;
        for (unsigned long j = BMT_first_nz_ptr->read_integer_from_arr(i + 1) - 1; j >= BMT_first_nz_ptr->read_integer_from_arr(i); j--)
        {
            map = (map << 1) | bit_map[j];
            if(j==0)
            {
                break;
            }
        }
        thread_bit_map_vec.push_back(map);
    }

    assert(thread_bit_map_vec.size() == BMT_first_nz_ptr->get_len() - 1);


    shared_ptr<universal_array> thread_bit_map_ptr(new universal_array(&(thread_bit_map_vec[0]), thread_bit_map_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> thread_bit_map_item(new meta_data_item(thread_bit_map_ptr, THREAD_META, "thread_bit_map", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(thread_bit_map_item);

    shared_ptr<data_item_record> thread_bit_map_record(new data_item_record(THREAD_META, "thread_bit_map", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(thread_bit_map_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> thread_bit_map::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> thread_bit_map::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string thread_bit_map::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "thread_bit_map::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + "}";

    return return_str;
}