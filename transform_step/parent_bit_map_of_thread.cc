#include "data_transform_step.hpp"

parent_bit_map_of_thread::parent_bit_map_of_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int target_matrix_id)
    : basic_data_transform_step("parent_bit_map_of_thread", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
    this->pos = pos;
}

void parent_bit_map_of_thread::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);


        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices_without_ending", this->target_matrix_id));

    }

    shared_ptr<data_item_record> first_row_indices_record(new data_item_record(THREAD_META, "first_row_indices_without_ending", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_row_indices_record);

    shared_ptr<universal_array> BMT_first_row_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_row_indices_without_ending", this->target_matrix_id)->get_metadata_arr();

    vector<bool> bit_map;
    bit_map.push_back(true);
    for (unsigned long j = 1; j < BMT_first_row_ptr->get_len(); j++)
    {
        unsigned long cur_row = BMT_first_row_ptr->read_integer_from_arr(j);
        unsigned long former_row = BMT_first_row_ptr->read_integer_from_arr(j - 1);

        if (cur_row != former_row)
        {
            bit_map.push_back(true);
        }
        else
        {
            bit_map.push_back(false);
        }
    }
    assert(bit_map.size() == BMT_first_row_ptr->get_len());

    if (this->pos == WARP_META)
    {
        for (int i = 0; i < bit_map.size(); i += get_config()["VECTOR_WIDTH"].as_float())
        {
            bit_map[i] = true;
        }
    }
    else if (this->pos == TBLOCK_META)
    {
        shared_ptr<universal_array> first_BMT_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_BMT_indices", this->target_matrix_id)->get_metadata_arr();
        for(int i=0;i<first_BMT_ptr->get_len();i++)
        {
            unsigned long cur_BMT = first_BMT_ptr->read_integer_from_arr(i);
            bit_map[cur_BMT] = true;
        }
    }

    vector<unsigned long> parent_bit_map_vec;
    unsigned int warp_size = get_config()["VECTOR_WIDTH"].as_float();
    unsigned int count = 0;
    if (this->pos == WARP_META)
    {
        for (unsigned long i = 0; i < BMT_first_row_ptr->get_len(); i += warp_size)
        {
            unsigned long map = 0;
            unsigned long k = (i + warp_size - 1) > (BMT_first_row_ptr->get_len() - 1) ? (BMT_first_row_ptr->get_len() - 1) : (i + warp_size - 1);

            for (unsigned int j = k; j >= i; j --)
            {
                count += 1;
                map = (map << 1) | bit_map[j];
                if(j == 0)
                {
                    break;
                }
            }
            parent_bit_map_vec.push_back(map);
        }
    }
    else if (this->pos == TBLOCK_META)
    {
        this->pos = THREAD_META;
        for (unsigned long i = 0; i < bit_map.size(); i++)
        {
            parent_bit_map_vec.push_back(bit_map[i]);
        }
    }


    shared_ptr<universal_array> parent_bit_map_ptr(new universal_array(&(parent_bit_map_vec[0]), parent_bit_map_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> parent_bit_map_item(new meta_data_item(parent_bit_map_ptr, this->pos, "bit_map_of_thread", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(parent_bit_map_item);

    shared_ptr<data_item_record> parent_bit_map_record(new data_item_record(this->pos, "bit_map_of_thread", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(parent_bit_map_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> parent_bit_map_of_thread::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> parent_bit_map_of_thread::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string parent_bit_map_of_thread::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "parent_bit_map_of_thread::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + "}";

    return return_str;
}