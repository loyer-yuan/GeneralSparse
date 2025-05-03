#include "../data_transform_step.hpp"

modify_vals_by_col_pad_in_sub_matrix::modify_vals_by_col_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple_of_each_row_size)
    : basic_data_transform_step("modify_vals_by_col_pad_in_sub_matrix", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(multiple_of_each_row_size >= 2);

    this->multiple_of_each_row_size = multiple_of_each_row_size;
    this->target_matrix_id = target_matrix_id;
}

void modify_vals_by_col_pad_in_sub_matrix::run(bool check)
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
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
    }

    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> nz_vals_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 当前的数据类型
    data_type cur_type = nz_vals_ptr->get_data_type();

    if (check)
    {
        assert(end_row_index >= start_row_index);
        assert(nz_row_indices_ptr->get_len() == nz_vals_ptr->get_len());
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

    // 行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = nz_row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;

    vector<vector<double>> val_of_each_row(row_num);

    // 将值放到按行分割的二维数组
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        // 当前非零元的行索引
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);
        double cur_val = nz_vals_ptr->read_float_from_arr(i);
        if (check)
        {
            assert(cur_row_index < row_num);
        }
        // 将非零元索引放在对应桶中
        val_of_each_row[cur_row_index].push_back(cur_val);
    }

    bool is_padded = false;

    // 执行padding
    for (unsigned long i = 0; i < val_of_each_row.size(); i++)
    {
        unsigned long cur_row_size = val_of_each_row[i].size();

        // 需要padding
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
                    cout << "modify_vals_by_col_pad_in_sub_matrix::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                    assert(false);
                }
            }

            for (unsigned long j = cur_row_size; j < target_row_size; j++)
            {
                val_of_each_row[i].push_back(0);
                is_padded = true;
            }
        }
    }

    // padding了
    if (is_padded == true)
    {
        shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

        shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(begin_row_index_record);

        shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(end_row_index_record);

        shared_ptr<data_item_record> nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(nz_vals_record);

        // 将所有值拉成一个直线
        vector<double> new_val_vec;

        // 遍历二维数组
        for (unsigned long i = 0; i < val_of_each_row.size(); i++)
        {
            for (unsigned long j = 0; j < val_of_each_row[i].size(); j++)
            {
                new_val_vec.push_back(val_of_each_row[i][j]);
            }
        }

        shared_ptr<data_item_record> new_nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->dest_data_item_ptr_vec.push_back(nz_vals_record);

        // 删除值数组
        this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

        // 将na_vals插入到metadata set中
        shared_ptr<universal_array> new_row_index_ptr(new universal_array(&(new_val_vec[0]), new_val_vec.size(), DOUBLE));

        if (cur_type == FLOAT)
        {
            new_row_index_ptr->compress_float_precise();
            if (check)
            {
                assert(new_row_index_ptr->get_data_type() == FLOAT);
            }
        }

        shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_row_index_ptr, GLOBAL_META, "nz_vals", this->target_matrix_id));

        this->meta_data_set_ptr->add_element(item_ptr);
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_vals_by_col_pad_in_sub_matrix::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> modify_vals_by_col_pad_in_sub_matrix::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->multiple_of_each_row_size >= 2);
    assert(this->is_run == true);

    // 空指针检查
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_vals_by_col_pad_in_sub_matrix::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->multiple_of_each_row_size >= 2);

    string return_str = "modify_vals_by_col_pad_in_sub_matrix::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",multiple_of_each_row_size:" + to_string(this->multiple_of_each_row_size) + "}";

    return return_str;
}