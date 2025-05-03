#include "../data_transform_step.hpp"

get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB::get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size)
    : basic_data_transform_step("get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(col_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_col_block_size = col_size;
}

void get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->fixed_col_block_size > 0);
        // 检查
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());

        // 查看当前子块的边界
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 非零元行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 行索引，得到每一行的非零元数量，从而得到分块点
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 取出所有的BMTB行偏移量
    shared_ptr<universal_array> BMTB_first_row_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 计入输入数据
    shared_ptr<data_item_record> start_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(start_row_index_record);
    shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(end_row_index_record);
    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    // 记录取出的BMTB行偏移
    shared_ptr<data_item_record> BMTB_first_row_indices_record(new data_item_record(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(BMTB_first_row_indices_record);

    if (check)
    {
        assert(end_row_index >= start_row_index);
    }

    // 查看真正的行结束位置
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
        // 查看当前上一个父块的类型
        bool BMTB_blocking_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);

        assert(BMTB_blocking_existing == true);

        // 之前的父块只有行切块
        if (BMTB_blocking_existing == true)
        {
            assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
        }

        // 获得每一行的行长度
        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> BMW_begin_row_vec_relative_to_BMTB;

    // 遍历所有的BMTB
    for (unsigned long i = 0; i < BMTB_first_row_indices_ptr->get_len() - 1; i++)
    {
        // BMTB的行边界
        unsigned long BMTB_first_row_index = BMTB_first_row_indices_ptr->read_integer_from_arr(i);
        unsigned long next_BMTB_first_row_index = BMTB_first_row_indices_ptr->read_integer_from_arr(i + 1);
        if (check)
        {
            assert(BMTB_first_row_index < next_BMTB_first_row_index);
        }

        // 遍历对应行
        for (unsigned long j = BMTB_first_row_index; j < next_BMTB_first_row_index; j++)
        {
            // 查看对应的行的长度
            unsigned long cur_row_size = nnz_of_each_row[j];

            // 当前行的BMW数量
            unsigned long BMW_num_of_row = cur_row_size / this->fixed_col_block_size;

            // 如果不能整除，那就加一个BMW
            if (cur_row_size % this->fixed_col_block_size != 0)
            {
                BMW_num_of_row = BMW_num_of_row + 1;
            }

            // 遍历所有的BMW，写每一个BMW的首行索引
            for (unsigned long BMW_id_of_row = 0; BMW_id_of_row < BMW_num_of_row; BMW_id_of_row++)
            {
                // 相对索引
                BMW_begin_row_vec_relative_to_BMTB.push_back(j - BMTB_first_row_index);
            }
        }
    }
    if (check)
    {
        assert(BMW_begin_row_vec_relative_to_BMTB.size() > 0);
    }

    // 将相对索引放到metadata set中
    shared_ptr<universal_array> BMW_first_row_index_relative_to_BMTB_ptr(new universal_array(&(BMW_begin_row_vec_relative_to_BMTB[0]), BMW_begin_row_vec_relative_to_BMTB.size(), UNSIGNED_LONG));
    // 行偏移的相对索引
    shared_ptr<meta_data_item> BMW_first_row_index_relative_to_BMTB_item(new meta_data_item(BMW_first_row_index_relative_to_BMTB_ptr, WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
    // 将新的内容写到metadata set中
    this->meta_data_set_ptr->add_element(BMW_first_row_index_relative_to_BMTB_item);

    // 执行记录
    shared_ptr<data_item_record> BMW_first_row_index_relative_to_BMTB_record(new data_item_record(WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMW_first_row_index_relative_to_BMTB_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB::get_source_data_item_ptr_in_data_transform_step()
{
    // 内容满足要求
    assert(target_matrix_id >= 0);
    assert(fixed_col_block_size > 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // 内容满足要求
    assert(target_matrix_id >= 0);
    assert(fixed_col_block_size > 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB::convert_to_string()
{
    assert(target_matrix_id >= 0);
    assert(fixed_col_block_size > 0);

    string return_str = "get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",col_size:" + to_string(this->fixed_col_block_size) + "}";

    return return_str;
}