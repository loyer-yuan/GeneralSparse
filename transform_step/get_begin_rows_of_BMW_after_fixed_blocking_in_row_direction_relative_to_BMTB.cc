#include "data_transform_step.hpp"

get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB::get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size)
    : basic_data_transform_step("get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_block_size = fixed_row_block_size;
}

void get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->fixed_row_block_size >= 1);
        // 检查
        assert(this->meta_data_set_ptr->check());

        // 存在TBLOCk级别的数据
        assert(this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id) > 0);

        // 需要当前子块的边界行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 查看有没有行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // tblock的行边界
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    }

    // 读出来当前子块行边界
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 当前的行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    // tblock行边界
    shared_ptr<universal_array> BMTB_first_row_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

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

    if (check)
    {
        assert(end_row_index >= start_row_index);
    }

    // 当前的行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    // 相对索引
    vector<unsigned long> nz_begin_index_of_each_BMW_relative_to_BMTB;

    // 遍历所有BMTB，在nz_begin_index_of_each_BMW_relative_to_BMTB中插入BMW和BMTB的行边界
    for (unsigned long i = 0; i < BMTB_first_row_indices_ptr->get_len() - 1; i++)
    {
        unsigned long BMTB_first_row_index = BMTB_first_row_indices_ptr->read_integer_from_arr(i);
        unsigned long next_BMTB_first_row_index = BMTB_first_row_indices_ptr->read_integer_from_arr(i + 1);

        // 填充行边界
        for (unsigned long BMW_first_row_index = BMTB_first_row_index; BMW_first_row_index < next_BMTB_first_row_index; BMW_first_row_index = BMW_first_row_index + this->fixed_row_block_size)
        {
            nz_begin_index_of_each_BMW_relative_to_BMTB.push_back(BMW_first_row_index - BMTB_first_row_index);
        }
    }

    // 将相对索引放到metadata set中
    shared_ptr<universal_array> BMW_first_row_index_relative_to_BMTB_ptr(new universal_array(&(nz_begin_index_of_each_BMW_relative_to_BMTB[0]), nz_begin_index_of_each_BMW_relative_to_BMTB.size(), UNSIGNED_LONG));
    // BMW行偏移的全局索引
    shared_ptr<meta_data_item> BMW_first_row_index_relative_to_BMTB_ptr_item_ptr(new meta_data_item(BMW_first_row_index_relative_to_BMTB_ptr, WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMW_first_row_index_relative_to_BMTB_ptr_item_ptr);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    vector<shared_ptr<data_item_record>> record_vec;

    // 需要读入行索引
    shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    record_vec.push_back(row_indices_record);

    // 只需要读入当前子矩阵的尾部的行边界
    shared_ptr<data_item_record> row_end_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    record_vec.push_back(row_end_record);

    // 行边界
    shared_ptr<data_item_record> row_start_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    record_vec.push_back(row_start_record);

    // BMTB行边界
    shared_ptr<data_item_record> BMTB_row_start_record(new data_item_record(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    record_vec.push_back(BMTB_row_start_record);

    return record_vec;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    vector<shared_ptr<data_item_record>> record_vec;

    shared_ptr<data_item_record> BMW_first_row_index_relative_to_BMTB_record(new data_item_record(WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
    record_vec.push_back(BMW_first_row_index_relative_to_BMTB_record);

    return record_vec;
}

// 转化为字符串
string get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    string return_str = "get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}