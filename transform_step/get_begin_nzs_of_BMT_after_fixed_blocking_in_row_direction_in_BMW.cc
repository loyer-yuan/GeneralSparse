#include "data_transform_step.hpp"

get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW::get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size)
    : basic_data_transform_step("get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_block_size = fixed_row_block_size;
}

void get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);
        assert(this->fixed_row_block_size >= 1);

        // 当前有WARP级别的数据
        assert(this->meta_data_set_ptr->count_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id) > 0);

        // 需要当前子块的边界行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 查看有没有行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        //查看warp边界
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id));

        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id));
    }

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(begin_row_index_record);

    shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(end_row_index_record);

    shared_ptr<data_item_record> first_row_indices_record(new data_item_record(WARP_META, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_row_indices_record);

    shared_ptr<data_item_record> first_nz_indices_record(new data_item_record(WARP_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_record);

    // 读出来当前子矩阵的边界行索引
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 当前的行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    shared_ptr<universal_array> BMW_first_row_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

    shared_ptr<universal_array> BMW_first_nzs_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

    if (check)
    {
        assert(end_row_index >= start_row_index);
    }

    // 真正的行结束边界
    unsigned long real_end_row_index = start_row_index + nz_row_indices_ptr->read_integer_from_arr(nz_row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }
    // 当前的行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= start_row_index);

        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> BMT_begin_nz_vec;

    // 遍历所有BMW
    for (unsigned long j = 0; j < BMW_first_row_indices_ptr->get_len() - 1; j++)
    {
        // 当前BMW的行偏移
        unsigned long BMW_first_row_index = BMW_first_row_indices_ptr->read_integer_from_arr(j);
        unsigned long next_BMW_first_row_index = BMW_first_row_indices_ptr->read_integer_from_arr(j + 1);

        // 当前BMW非零元偏移量
        unsigned long BMW_first_nz = BMW_first_nzs_ptr->read_integer_from_arr(j);
        unsigned long next_BMW_first_nz = BMW_first_nzs_ptr->read_integer_from_arr(j + 1);

        int row_count = 0;
        int nz_count = 0;

        BMT_begin_nz_vec.push_back(BMW_first_nz);

        // 遍历所有的行
        for (unsigned long i = BMW_first_row_index; i < next_BMW_first_row_index; i++)
        {
            unsigned long cur_row_size = nnz_of_each_row[i];

            row_count += 1;
            nz_count += cur_row_size;

            if (row_count == this->fixed_row_block_size && i != (next_BMW_first_row_index - 1))
            {
                BMT_begin_nz_vec.push_back(BMW_first_nz + nz_count);
                row_count = 0;
            }
        }
    }
    BMT_begin_nz_vec.push_back(BMW_first_nzs_ptr->read_integer_from_arr(BMW_first_nzs_ptr->get_len() - 1));

    // 将数据放到metadata set中
    shared_ptr<universal_array> BMT_first_nz_ptr(new universal_array(&(BMT_begin_nz_vec[0]), BMT_begin_nz_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMT_first_nz_item(new meta_data_item(BMT_first_nz_ptr, THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMT_first_nz_item);

    // 执行记录
    shared_ptr<data_item_record> BMT_first_nz_record(new data_item_record(THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMT_first_nz_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);
    assert(this->is_run == true);

    // 空指针检查
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    string return_str = "get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}