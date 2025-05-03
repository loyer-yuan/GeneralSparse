#include "data_transform_step.hpp"

get_begin_nzs_of_level_after_merge_path::get_begin_nzs_of_level_after_merge_path(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, POS_TYPE pos, int work_size)
    : basic_data_transform_step("get_begin_nzs_of_level_after_merge_path", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(work_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->work_size = work_size;
    this->pos = pos;
}

void get_begin_nzs_of_level_after_merge_path::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(begin_row_index_record);

    shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(end_row_index_record);

    if (check)
    {
        assert(end_row_index >= start_row_index);
    }
    unsigned long real_end_row_index = start_row_index + nz_row_indices_ptr->read_integer_from_arr(nz_row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }
    unsigned long row_num = end_row_index - start_row_index + 1;
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);
    
    if (check)
    {
        assert(end_row_index >= start_row_index);
        assert(nnz_of_each_row.size() == row_num);
    }



    vector<unsigned long> total_path;
    unsigned long count = 0;
    bool flag = true;


    for(unsigned long i = 0; i < row_num; i++)
    {

        if (nnz_of_each_row[i] != 0)
        {
            count += 1;
            if (flag == true)
            {
                flag = false;
                count -= 1;
            }
            count += nnz_of_each_row[i];
            total_path.push_back(count);
        }
    }


    vector<unsigned long> level_begin_nz_vec;

    for(unsigned long i = 0; i < count; i += this->work_size)
    {
        for(unsigned long j = 0; j < total_path.size(); j++)
        {
            if (total_path[j] > i)
            {
                level_begin_nz_vec.push_back(i - j);
                break;
            }
        }
    }
    level_begin_nz_vec.push_back(nz_row_indices_ptr->get_len());



    // 将数据放到metadata set中
    shared_ptr<universal_array> level_first_nz_ptr(new universal_array(&(level_begin_nz_vec[0]), level_begin_nz_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> level_first_nz_item(new meta_data_item(level_first_nz_ptr, pos, "first_nz_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(level_first_nz_item);

    // 执行记录
    shared_ptr<data_item_record> level_first_nz_record(new data_item_record(pos, "first_nz_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(level_first_nz_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_level_after_merge_path::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> get_begin_nzs_of_level_after_merge_path::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string get_begin_nzs_of_level_after_merge_path::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "get_begin_nzs_of_level_after_merge_path::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",work_size:" + to_string(this->work_size) + "}";

    return return_str;
}