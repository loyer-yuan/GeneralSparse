#include "data_transform_step.hpp"

get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size)
    : basic_data_transform_step("get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->fixed_row_block_size = fixed_row_block_size;
    this->target_matrix_id = target_matrix_id;
}

void get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);

        // 需要当前子块的边界行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 查看有没有行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

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
        cout << "get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::run(): detect row padding before BMTB" << endl;
        end_row_index = real_end_row_index;
    }
    if (check)
    {
        assert(end_row_index >= start_row_index);
    }

    // 当前的行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    // 用一个数组存储所有的行起始索引
    vector<unsigned long> row_begin_index_vec;
    // 第一行肯定是从0开始
    row_begin_index_vec.push_back(0);

    // 查看完整分块的个数
    int complete_block_num = row_num / this->fixed_row_block_size;

    for (int i = 0; i < complete_block_num; i++)
    {
        row_begin_index_vec.push_back((i + 1) * this->fixed_row_block_size);
        if (check)
        {
            assert(((i + 1) * this->fixed_row_block_size) <= row_num);
        }
    }

    if (row_num % this->fixed_row_block_size != 0)
    {
        row_begin_index_vec.push_back(row_num);
    }
    if (check)
    {
        assert(row_begin_index_vec.size() >= 2);
    }

    // 产生一个新的metadata item
    shared_ptr<universal_array> BMTB_begin_row_index_ptr(new universal_array(&(row_begin_index_vec[0]), row_begin_index_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> item_ptr(new meta_data_item(BMTB_begin_row_index_ptr, TBLOCK_META, "first_row_indices", this->target_matrix_id));

    // 加入元素
    this->meta_data_set_ptr->add_element(item_ptr);
    // 执行完成
    this->is_run = true;
}

// 输入只有当前子块的最后一个非零元的行索引的值
vector<shared_ptr<data_item_record>> get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

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

    return record_vec;
}

// 输出只有一个，是BMTB的每一行首行索引
vector<shared_ptr<data_item_record>> get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    vector<shared_ptr<data_item_record>> record_vec;

    // 只需要一个输出，每个BMTB的首行索引
    shared_ptr<data_item_record> BMTB_begin_row_index_record(new data_item_record(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    record_vec.push_back(BMTB_begin_row_index_record);

    return record_vec;
}

string get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}