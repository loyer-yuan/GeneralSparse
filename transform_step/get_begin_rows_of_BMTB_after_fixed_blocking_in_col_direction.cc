#include "../data_transform_step.hpp"

get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction::get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size)
    : basic_data_transform_step("get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(col_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_col_block_size = col_size;
}

void get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->fixed_col_block_size > 0);
        // 检查
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());

        // 不能出现交错存储
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == false);

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

    // 计入输入数据
    shared_ptr<data_item_record> start_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(start_row_index_record);
    shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(end_row_index_record);
    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

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
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id) == false);
        // 获得每一行的行长度
        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> BMTB_begin_row_vec;

    // 遍历所有的行，一个个分块，起始行为当前遍历的行号
    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        // 获取当前行的长度
        unsigned long cur_row_size = nnz_of_each_row[i];

        // 当前行的BMTB数量
        unsigned long BMTB_num_of_row = cur_row_size / this->fixed_col_block_size;

        // 如果不能整除，那就加一个BMTB
        if (cur_row_size % this->fixed_col_block_size != 0)
        {
            BMTB_num_of_row = BMTB_num_of_row + 1;
        }

        // 遍历所有的BMTB，写每一个BMTB的首行索引
        for (unsigned long BMTB_id_of_row = 0; BMTB_id_of_row < BMTB_num_of_row; BMTB_id_of_row++)
        {
            BMTB_begin_row_vec.push_back(i);
        }
    }

    // 将数据放到metadata set中
    shared_ptr<universal_array> BMTB_first_row_ptr(new universal_array(&(BMTB_begin_row_vec[0]), BMTB_begin_row_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMTB_first_row_item(new meta_data_item(BMTB_first_row_ptr, TBLOCK_META, "first_row_indices_without_ending", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMTB_first_row_item);

    // 执行记录
    shared_ptr<data_item_record> BMTB_first_row_without_end_record(new data_item_record(TBLOCK_META, "first_row_indices_without_ending", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMTB_first_row_without_end_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_col_block_size > 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_col_block_size > 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_col_block_size > 0);

    string return_str = "get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",col_size:" + to_string(this->fixed_col_block_size) + "}";

    return return_str;
}