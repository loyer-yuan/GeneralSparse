#include "../data_transform_step.hpp"

modify_row_indices_by_col_pad_in_sub_matrix::modify_row_indices_by_col_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple_of_each_row_size)
    : basic_data_transform_step("modify_row_indices_by_col_pad_in_sub_matrix", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(multiple_of_each_row_size >= 2);

    this->multiple_of_each_row_size = multiple_of_each_row_size;
    this->target_matrix_id = target_matrix_id;
}

// 执行列方向的padding
void modify_row_indices_by_col_pad_in_sub_matrix::run(bool check)
{
    if (check)
    {
        // 保证metadata set
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        // 保证矩阵号的正确
        assert(this->target_matrix_id >= 0);
        assert(this->multiple_of_each_row_size >= 2);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
    }

    // 查看当前子块的的最后一行
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    assert(end_row_index >= start_row_index);

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

    // 查看行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = nz_row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;

    // 用一个数组，将所有的行索引，放到一个二维数组中
    vector<vector<unsigned long>> row_index_of_each_row(row_num);

    // 遍历所有的行索引，将行索引执行分桶
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        // 当前非零元的行索引
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);

        if (check)
        {
            assert(cur_row_index < row_num);
        }
        // 将非零元索引放在对应桶中
        row_index_of_each_row[cur_row_index].push_back(cur_row_index);
    }

    bool is_padded = false;

    // 遍历每一个桶，根据当前桶的大小执行padding，padding到multiple_of_each_row_size的整数倍
    for (unsigned long i = 0; i < row_index_of_each_row.size(); i++)
    {
        // 当前行长度
        unsigned long cur_row_size = row_index_of_each_row[i].size();
        // 目标行长度
        if (cur_row_size % this->multiple_of_each_row_size != 0)
        {
            unsigned long target_row_size = (cur_row_size / this->multiple_of_each_row_size + 1) * this->multiple_of_each_row_size;

            if (check)
            {
                assert(target_row_size > cur_row_size);
            }

            unsigned long added_row_nnz = target_row_size - cur_row_size;

            nnz_after_padding = nnz_after_padding + added_row_nnz;

            if (check)
            {
                // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                {
                    cout << "modify_row_indices_by_col_pad_in_sub_matrix::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                    assert(false);
                }
            }

            for (unsigned long j = cur_row_size; j < target_row_size; j++)
            {
                row_index_of_each_row[i].push_back(row_index_of_each_row[i][row_index_of_each_row[i].size() - 1]);
                is_padded = true;
            }
        }
    }

    // 这个transform是需要执行的
    if (is_padded == true)
    {
        shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

        shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(begin_row_index_record);

        shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(end_row_index_record);

        // 将所有的行索引拉成一个直线
        vector<unsigned long> new_row_index_vec;

        for (unsigned long i = 0; i < row_index_of_each_row.size(); i++)
        {
            for (unsigned long j = 0; j < row_index_of_each_row[i].size(); j++)
            {
                new_row_index_vec.push_back(row_index_of_each_row[i][j]);
            }
        }

        shared_ptr<data_item_record> new_nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->dest_data_item_ptr_vec.push_back(new_nz_row_indices_record);

        // 删除对应的nz_row_indices
        this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);

        // 将new_row_index_vec转化为通用数组
        shared_ptr<universal_array> new_row_index_ptr(new universal_array(&(new_row_index_vec[0]), new_row_index_vec.size(), UNSIGNED_LONG));
        shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_row_index_ptr, GLOBAL_META, "nz_row_indices", this->target_matrix_id));

        this->meta_data_set_ptr->add_element(item_ptr);
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_row_indices_by_col_pad_in_sub_matrix::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->multiple_of_each_row_size >= 2);
    assert(this->is_run == true);

    // 空指针检查
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_row_indices_by_col_pad_in_sub_matrix::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->multiple_of_each_row_size >= 2);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_row_indices_by_col_pad_in_sub_matrix::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->multiple_of_each_row_size >= 2);

    string return_str = "modify_row_indices_by_col_pad_in_sub_matrix::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",multiple_of_each_row_size:" + to_string(this->multiple_of_each_row_size) + "}";

    return return_str;
}