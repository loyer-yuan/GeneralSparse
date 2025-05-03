#include "data_transform_step.hpp"

get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction::get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size)
    : basic_data_transform_step("get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_block_size = fixed_row_block_size;
}

// 使用行索引
void get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);

        // 当前子块的起始行索引和结束行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 当前子块所有非零元的行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    // 获取当前子矩阵结束的行索引
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
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
    // 查看行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    // 创造一个数组，记录每一行的非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= start_row_index);
        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> nz_begin_index_of_each_BMTB;
    nz_begin_index_of_each_BMTB.push_back(0);

    // 遍历每一行的非零元，得出每个子块的非零元数量，加到一起得到偏移量
    unsigned long BMTB_num = row_num / this->fixed_row_block_size;

    // 如果不能整除就多一个
    if (row_num % this->fixed_row_block_size != 0)
    {
        BMTB_num = BMTB_num + 1;
    }

    // 遍历每一个块
    for (unsigned long i = 0; i < BMTB_num; i++)
    {
        // 当前快的行其实位置和结束位置
        unsigned long BMTB_first_row_index = i * this->fixed_row_block_size;
        unsigned long next_BMTB_first_row_index = (i + 1) * this->fixed_row_block_size;

        unsigned long cur_nz_num = 0;

        for (unsigned long row_id = BMTB_first_row_index; row_id < next_BMTB_first_row_index && row_id < nnz_of_each_row.size(); row_id++)
        {
            cur_nz_num = cur_nz_num + nnz_of_each_row[row_id];
        }

        nz_begin_index_of_each_BMTB.push_back(cur_nz_num + nz_begin_index_of_each_BMTB[nz_begin_index_of_each_BMTB.size() - 1]);
    }

    // 最后一位
    if (check)
    {
        assert(nz_begin_index_of_each_BMTB[nz_begin_index_of_each_BMTB.size() - 1] == nz_row_indices_ptr->get_len());
        assert(nz_begin_index_of_each_BMTB.size() >= 2);
    }

    // nz_begin_index_of_each_BMTB写回
    shared_ptr<universal_array> BMTB_begin_nz_index_ptr(new universal_array(&(nz_begin_index_of_each_BMTB[0]), nz_begin_index_of_each_BMTB.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> item_ptr(new meta_data_item(BMTB_begin_nz_index_ptr, TBLOCK_META, "first_nz_indices", this->target_matrix_id));

    // 加入元素
    this->meta_data_set_ptr->add_element(item_ptr);

    this->is_run = true;
}

// 输入只有当前子块的最后一个非零元的行索引的值
vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);

    vector<shared_ptr<data_item_record>> record_vec;

    // 两个行边界
    shared_ptr<data_item_record> row_end_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    record_vec.push_back(row_end_record);
    shared_ptr<data_item_record> row_start_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    record_vec.push_back(row_start_record);

    // 一个行索引
    shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    record_vec.push_back(row_indices_record);

    return record_vec;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    vector<shared_ptr<data_item_record>> record_vec;

    shared_ptr<data_item_record> first_nz_indices_record(new data_item_record(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    record_vec.push_back(first_nz_indices_record);

    return record_vec;
}

string get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}