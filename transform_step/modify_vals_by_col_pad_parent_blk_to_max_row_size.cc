#include "../data_transform_step.hpp"

modify_vals_by_col_pad_parent_blk_to_max_row_size::modify_vals_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool padding_with_empty_row)
    : basic_data_transform_step("modify_vals_by_col_pad_parent_blk_to_max_row_size", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);
    this->parent_pos = parent_pos;
    this->target_matrix_id = target_matrix_id;
    this->padding_with_empty_row = padding_with_empty_row;
}

void modify_vals_by_col_pad_parent_blk_to_max_row_size::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(meta_data_set_ptr != NULL);
        assert(meta_data_set_ptr->check());
        assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

        // 使用全局的行索引来执行padding，padding实际上是坐在列索引上
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 存在对应行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    // 查看当前子块的行边界
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 行索引
    shared_ptr<universal_array> row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    unsigned long last_row_index = row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    // 值
    shared_ptr<universal_array> vals_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

    // 当前行索引还没有padding过
    if (check)
    {
        assert(row_indices_ptr->get_len() == vals_ptr->get_len());
    }

    // 之前行方向可能padding过，所以要看看实际上的行数量
    // 真正的行结束边界
    unsigned long real_end_row_index = begin_row_index + row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }
    // 查看行数量
    unsigned long row_num = end_row_index - begin_row_index + 1;

    // 获得每一行的非零元数量，
    vector<unsigned long> row_nz_number = get_nnz_of_each_row_in_spec_range(row_indices_ptr, 0, row_num - 1, 0, row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= begin_row_index);
        assert(row_nz_number.size() == row_num);
    }

    // 将所有值放到对应二维数组中，方便进行统计和padding
    vector<vector<double>> vals_of_each_row(row_num);

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;

    // 遍历所有的值
    for (unsigned long i = 0; i < vals_ptr->get_len(); i++)
    {
        // 获取当前非零元的行索引。
        unsigned long cur_row_index = row_indices_ptr->read_integer_from_arr(i);
        if (check)
        {
            assert(cur_row_index < vals_of_each_row.size());
        }

        // 获取当前的值
        double cur_val = vals_ptr->read_float_from_arr(i);
        // assert(cur_val != 0);
        vals_of_each_row[cur_row_index].push_back(cur_val);
    }

    // 针对全局和局部的padding不一样，全局的padding单独处理
    if (parent_pos == GLOBAL_META)
    {
        // 查看最长的行
        unsigned long max_row_length = *max_element(row_nz_number.begin(), row_nz_number.end());

        // 查看是不是被padding了
        bool is_padding = false;

        // 遍历所有的行
        for (unsigned long i = 0; i < vals_of_each_row.size(); i++)
        {
            // 当前行的非零元数量
            unsigned long cur_row_size = vals_of_each_row[i].size();
            if (check)
            {
                assert(cur_row_size <= max_row_length);
                assert(cur_row_size == row_nz_number[i]);
            }
            if (this->padding_with_empty_row == true)
            {
                if (max_row_length != 0 && cur_row_size != max_row_length)
                {
                    unsigned long target_row_length = max_row_length;

                    // 要增加的非零元数量
                    if (check)
                    {
                        assert(target_row_length > cur_row_size);
                    }
                    unsigned long added_row_nz_num = target_row_length - cur_row_size;

                    nnz_after_padding = nnz_after_padding + added_row_nz_num;
                    if (check)
                    {
                        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                        {
                            cout << "modify_vals_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                            assert(false);
                        }
                    }
                    // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存

                    // 增加内容，添加0元
                    for (unsigned long added_nz = 0; added_nz < added_row_nz_num; added_nz++)
                    {
                        vals_of_each_row[i].push_back(0);
                    }

                    is_padding = true;

                    cur_row_size = vals_of_each_row[i].size();
                    if (check)
                    {
                        assert(cur_row_size % max_row_length == 0);
                    }
                }
            }
            else
            {
                if (max_row_length != 0 && cur_row_size % max_row_length != 0)
                {
                    unsigned long target_row_length = max_row_length;

                    // 要增加的非零元数量
                    if (check)
                    {
                        assert(target_row_length > cur_row_size);
                    }
                    unsigned long added_row_nz_num = target_row_length - cur_row_size;

                    nnz_after_padding = nnz_after_padding + added_row_nz_num;
                    if (check)
                    {
                        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                        {
                            cout << "modify_vals_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                            assert(false);
                        }
                    }
                    // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存

                    // 增加内容，添加0元
                    for (unsigned long added_nz = 0; added_nz < added_row_nz_num; added_nz++)
                    {
                        vals_of_each_row[i].push_back(0);
                    }

                    is_padding = true;

                    cur_row_size = vals_of_each_row[i].size();
                    if (check)
                    {
                        assert(cur_row_size % max_row_length == 0);
                    }
                }
            }

            // 完成了padding
        }

        // 如果真的padding了就记录输入
        if (is_padding == true)
        {
            // 记录当前读取的数据
            shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(row_indices_record);
            shared_ptr<data_item_record> nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(nz_vals_record);
            shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(begin_row_index_record);
            shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(end_row_index_record);

            // 将数组拉平
            vector<double> vals_vec;

            // 遍历整个二维数组
            for (unsigned long i = 0; i < vals_of_each_row.size(); i++)
            {
                for (unsigned long j = 0; j < vals_of_each_row[i].size(); j++)
                {
                    vals_vec.push_back(vals_of_each_row[i][j]);
                }
            }

            // padding之后非零元一定变多了
            if (check)
            {
                assert(vals_vec.size() > row_indices_ptr->get_len());
            }

            // 记录当前的输出
            shared_ptr<data_item_record> new_nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
            this->dest_data_item_ptr_vec.push_back(new_nz_vals_record);

            data_type old_data_type = vals_ptr->get_data_type();

            // 删除之前的nz_vals
            this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

            // 将vals_vec转化为通用数组，然后存到metadata set中
            shared_ptr<universal_array> new_val_ptr(new universal_array(&(vals_vec[0]), vals_vec.size(), DOUBLE));

            if (old_data_type == FLOAT)
            {
                new_val_ptr->compress_float_precise();
                if (check)
                {
                    assert(new_val_ptr->get_data_type() == FLOAT);
                }
            }

            shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_val_ptr, GLOBAL_META, "nz_vals", this->target_matrix_id));

            this->meta_data_set_ptr->add_element(item_ptr);
        }
    }
    else if (parent_pos == TBLOCK_META || parent_pos == WARP_META)
    {
        // 读入父块的行边界来，并且通过其行偏移量索引来处理每一个子块的内容
        // 父块的行偏移量索引
        if (check)
        {
            assert(this->meta_data_set_ptr->is_exist(parent_pos, "first_row_indices", this->target_matrix_id));
        }
        shared_ptr<universal_array> first_row_indices_of_parent_ptr = this->meta_data_set_ptr->get_element(parent_pos, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

        // 查看是不是需要padding
        bool is_padding = false;

        // 遍历所有父块
        for (unsigned long i = 0; i < first_row_indices_of_parent_ptr->get_len() - 1; i++)
        {
            // 获得当前父块的行索引范围
            unsigned long first_row_index_of_cur_parent = first_row_indices_of_parent_ptr->read_integer_from_arr(i);
            unsigned long first_row_index_of_next_parent = first_row_indices_of_parent_ptr->read_integer_from_arr(i + 1);
            if (check)
            {
                assert(i + 1 < first_row_indices_of_parent_ptr->get_len());
                assert(first_row_index_of_next_parent >= first_row_index_of_cur_parent);
            }

            // 记录当前父块中最大的行长度
            unsigned long max_row_size_of_this_parent = 0;

            // 遍历当前区间的行非零元，找到最大的行长度
            for (unsigned long row_index = first_row_index_of_cur_parent; row_index < first_row_index_of_next_parent; row_index++)
            {
                unsigned long cur_row_size = vals_of_each_row[row_index].size();
                if (check)
                {
                    assert(cur_row_size == row_nz_number[row_index]);
                }

                if (max_row_size_of_this_parent < cur_row_size)
                {
                    max_row_size_of_this_parent = cur_row_size;
                }
            }

            // 遍历当前父块的行区间，将每一行padding到对应的大小
            for (unsigned long row_index = first_row_index_of_cur_parent; row_index < first_row_index_of_next_parent; row_index++)
            {
                unsigned long cur_row_size = vals_of_each_row[row_index].size();
                if (check)
                {
                    assert(row_index < vals_of_each_row.size());
                    assert(cur_row_size == row_nz_number[row_index]);
                    assert(cur_row_size <= max_row_size_of_this_parent);
                }

                if (this->padding_with_empty_row == true)
                {
                    if (max_row_size_of_this_parent != 0 && cur_row_size != max_row_size_of_this_parent)
                    {
                        unsigned long target_row_length = max_row_size_of_this_parent;

                        // 增加非零元数量
                        if (check)
                        {
                            assert(target_row_length > cur_row_size);
                        }
                        unsigned long added_row_nnz = target_row_length - cur_row_size;

                        nnz_after_padding = nnz_after_padding + added_row_nnz;

                        if (check)
                        {
                            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                            {
                                cout << "modify_vals_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                                assert(false);
                            }
                        }

                        // 要添加的内容是一个零元
                        for (unsigned long added_nz = 0; added_nz < added_row_nnz; added_nz++)
                        {
                            vals_of_each_row[row_index].push_back(0);
                        }

                        is_padding = true;

                        cur_row_size = vals_of_each_row[row_index].size();
                        if (check)
                        {
                            // 当前满足要求
                            if (cur_row_size % max_row_size_of_this_parent != 0)
                            {
                                cout << "modify_vals_by_col_pad_parent_blk_to_max_row_size::run(): invalid padding, cur_row_size:" << cur_row_size << ", max_row_size_of_this_parent:" << max_row_size_of_this_parent << ", row_index:" << row_index << ", parent_blk_id:" << i << endl;
                                assert(cur_row_size % max_row_size_of_this_parent == 0);
                            }
                        }
                    }
                }
                else
                {
                    if (max_row_size_of_this_parent != 0 && cur_row_size % max_row_size_of_this_parent != 0)
                    {
                        unsigned long target_row_length = max_row_size_of_this_parent;

                        // 增加非零元数量
                        if (check)
                        {
                            assert(target_row_length > cur_row_size);
                        }
                        unsigned long added_row_nnz = target_row_length - cur_row_size;

                        nnz_after_padding = nnz_after_padding + added_row_nnz;

                        if (check)
                        {
                            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                            {
                                cout << "modify_vals_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                                assert(false);
                            }
                        }

                        // 要添加的内容是一个零元
                        for (unsigned long added_nz = 0; added_nz < added_row_nnz; added_nz++)
                        {
                            vals_of_each_row[row_index].push_back(0);
                        }

                        is_padding = true;

                        cur_row_size = vals_of_each_row[row_index].size();
                        if (check)
                        {
                            // 当前满足要求
                            if (cur_row_size % max_row_size_of_this_parent != 0)
                            {
                                cout << "modify_vals_by_col_pad_parent_blk_to_max_row_size::run(): invalid padding, cur_row_size:" << cur_row_size << ", max_row_size_of_this_parent:" << max_row_size_of_this_parent << ", row_index:" << row_index << ", parent_blk_id:" << i << endl;
                                assert(cur_row_size % max_row_size_of_this_parent == 0);
                            }
                        }
                    }
                }
                // 查看当前行是不是需要padding的
            }
        }

        // 如果完成了padding，就将内容放到metadata set中
        if (is_padding == true)
        {
            // 记录当前读取的数据
            shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(row_indices_record);
            shared_ptr<data_item_record> nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(nz_vals_record);
            shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(begin_row_index_record);
            shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(end_row_index_record);
            shared_ptr<data_item_record> first_row_indices_record(new data_item_record(parent_pos, "first_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(first_row_indices_record);

            // 将数组拉平
            vector<double> vals_vec;

            // 遍历整个二维数组
            for (unsigned long i = 0; i < vals_of_each_row.size(); i++)
            {
                for (unsigned long j = 0; j < vals_of_each_row[i].size(); j++)
                {
                    vals_vec.push_back(vals_of_each_row[i][j]);
                }
            }

            // padding之后非零元一定变多了
            if (check)
            {
                assert(vals_vec.size() > row_indices_ptr->get_len());
            }

            // 记录当前的输出
            shared_ptr<data_item_record> new_nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
            this->dest_data_item_ptr_vec.push_back(new_nz_vals_record);

            data_type old_data_type = vals_ptr->get_data_type();

            // 删除之前的nz_vals
            this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

            // 将vals_vec转化为通用数组，然后存到metadata set中
            shared_ptr<universal_array> new_val_ptr(new universal_array(&(vals_vec[0]), vals_vec.size(), DOUBLE));

            if (old_data_type == FLOAT)
            {
                new_val_ptr->compress_float_precise();
                if (check)
                {
                    assert(new_val_ptr->get_data_type() == FLOAT);
                }
            }

            shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_val_ptr, GLOBAL_META, "nz_vals", this->target_matrix_id));

            this->meta_data_set_ptr->add_element(item_ptr);
        }
    }

    this->is_run = true;
}

// 查看当前step的输入
vector<shared_ptr<data_item_record>> modify_vals_by_col_pad_parent_blk_to_max_row_size::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    // 空指针检查
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_vals_by_col_pad_parent_blk_to_max_row_size::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_vals_by_col_pad_parent_blk_to_max_row_size::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    string return_str = "modify_vals_by_col_pad_parent_blk_to_max_row_size::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",parent_pos:" + convert_pos_type_to_string(parent_pos) + "}";

    return return_str;
}