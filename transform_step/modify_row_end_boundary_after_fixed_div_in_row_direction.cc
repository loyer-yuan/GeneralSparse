#include "data_transform_step.hpp"

modify_row_end_boundary_after_fixed_div_in_row_direction::modify_row_end_boundary_after_fixed_div_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size)
    : basic_data_transform_step("modify_row_end_boundary_after_fixed_div_in_row_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_gap_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_gap_size = fixed_row_gap_size;
}

void modify_row_end_boundary_after_fixed_div_in_row_direction::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->fixed_row_gap_size > 0);

        // 目标子矩阵的行边界都是存在的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 把行索读出来，发现潜在的空条带
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    // 读出来当前子块的首行索引和最后一行的索引
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 输入数据的记录
    shared_ptr<data_item_record> begin_row_index_record_ptr(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    shared_ptr<data_item_record> end_row_index_record_ptr(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(begin_row_index_record_ptr);
    this->source_data_item_ptr_vec.push_back(end_row_index_record_ptr);

    if (check)
    {
        assert(end_row_index >= begin_row_index);
    }

    // 查看当前行的数量
    unsigned long row_num = end_row_index - begin_row_index + 1;

    // 查看当前的行数量要分多少个子块
    unsigned long sub_matrix_num = row_num / this->get_fixed_row_gap_size();

    // 如果不能整除，那么就需要多一个子块
    if (row_num % this->get_fixed_row_gap_size() != 0)
    {
        sub_matrix_num = sub_matrix_num + 1;
    }

    // 读出行索引，查看空白行
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录行索引
    shared_ptr<data_item_record> nz_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record_ptr);

    // 用一个数组记录空桶
    vector<bool> empty_bin_flag(sub_matrix_num);

    for (unsigned long i = 0; i < sub_matrix_num; i++)
    {
        empty_bin_flag[i] = true;
    }

    // 遍历所有行数据
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);

        unsigned long cur_sub_matrix_id = cur_row_index / this->get_fixed_row_gap_size();

        if (check)
        {
            assert(cur_sub_matrix_id < sub_matrix_num);
        }

        empty_bin_flag[cur_sub_matrix_id] = false;
    }

    for (unsigned long i = 0; i < empty_bin_flag.size(); i++)
    {
        if (empty_bin_flag[i] == false)
        {
            int max_existing_sub_matrix_id_of_end_row_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "end_row_index");
            unsigned long new_end_row_index = begin_row_index + (i + 1) * this->fixed_row_gap_size - 1;

            // 超过的被分割的，取回来
            if (new_end_row_index >= end_row_index)
            {
                new_end_row_index = end_row_index;
            }

            shared_ptr<meta_data_item> end_row_index(new meta_data_item(((void *)(&new_end_row_index)), UNSIGNED_LONG, GLOBAL_META, "end_row_index", max_existing_sub_matrix_id_of_end_row_index + 1));
            if (check)
            {
                assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", max_existing_sub_matrix_id_of_end_row_index + 1) == false);
            }
            this->meta_data_set_ptr->add_element(end_row_index);

            shared_ptr<data_item_record> new_end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", max_existing_sub_matrix_id_of_end_row_index + 1));
            this->dest_data_item_ptr_vec.push_back(new_end_row_index_record);
        }
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_row_end_boundary_after_fixed_div_in_row_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_gap_size > 0);
    assert(this->is_run == true);

    // 检查输出数据
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(source_data_item_ptr_vec[i] != NULL);
    }

    // 返回输出数据
    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_row_end_boundary_after_fixed_div_in_row_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_gap_size > 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

// 转化为字符串
string modify_row_end_boundary_after_fixed_div_in_row_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_gap_size > 0);

    // 打印参数
    string return_str = "modify_row_end_boundary_after_fixed_div_in_row_direction::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",fixed_row_gap_size:" + to_string(this->fixed_row_gap_size) + "}";

    return return_str;
}