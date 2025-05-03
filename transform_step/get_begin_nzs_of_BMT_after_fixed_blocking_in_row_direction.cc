#include "data_transform_step.hpp"

get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction::get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size)
    : basic_data_transform_step("get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_block_size = fixed_row_block_size;
}

void get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);
        assert(this->fixed_row_block_size >= 1);

        // 当前没有TBLOCK、WARP级别的数据
        assert(this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id) == 0);
        assert(this->meta_data_set_ptr->count_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id) == 0);

        // 需要当前子块的边界行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 查看有没有行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(begin_row_index_record);

    shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(end_row_index_record);

    // 读出来当前子矩阵的边界行索引
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 当前的行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

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

    // 获得每一行的行长度
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= start_row_index);
        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> BMT_begin_nz_vec;
    BMT_begin_nz_vec.push_back(0);
    int row_count = 0;
    int nz_count = 0;

    // 遍历每一行，查看对应的BMT的非零元偏移量，BMT原则上不存在空的
    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        unsigned long cur_row_size = nnz_of_each_row[i];

        row_count += 1;
        nz_count += cur_row_size;

        if (row_count == this->fixed_row_block_size)
        {
            BMT_begin_nz_vec.push_back(nz_count);
            row_count = 0;
        }
    }

    if (row_count != this->fixed_row_block_size && row_count != 0)
    {
        BMT_begin_nz_vec.push_back(nz_count);
    }

    // 到这里，BMT_begin_nz_vec的最后一位是总非零元数量
    if (check)
    {
        assert(nz_row_indices_ptr->get_len() == BMT_begin_nz_vec[BMT_begin_nz_vec.size() - 1]);
    }

    // 将新的内容放到metadata set中
    shared_ptr<universal_array> BMT_begin_nz_ptr(new universal_array(&(BMT_begin_nz_vec[0]), BMT_begin_nz_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMT_begin_nz_item(new meta_data_item(BMT_begin_nz_ptr, THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMT_begin_nz_item);

    // 加入新的record
    shared_ptr<data_item_record> BMT_begin_nz_record(new data_item_record(THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMT_begin_nz_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    string return_str = "get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}