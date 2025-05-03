#include "../data_transform_step.hpp"

modify_col_indices_by_col_pad_parent_blk_to_max_row_size::modify_col_indices_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool padding_with_empty_row)
    : basic_data_transform_step("modify_col_indices_by_col_pad_parent_blk_to_max_row_size", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    this->parent_pos = parent_pos;
    this->target_matrix_id = target_matrix_id;
    this->padding_with_empty_row = padding_with_empty_row;
}

void modify_col_indices_by_col_pad_parent_blk_to_max_row_size::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(meta_data_set_ptr != NULL);
        assert(meta_data_set_ptr->check());
        assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);
        // 使用全局的行索引来执行padding，padding实际上是坐在列索引上
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
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

    // 列索引
    shared_ptr<universal_array> col_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();

    if (check)
    {
        // 当前行索引还没有padding过
        assert(row_indices_ptr->get_len() == col_indices_ptr->get_len());
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

    // 将所有列索引放到对应二维数组中，方便进行统计和padding
    vector<unsigned long> col_indices_of_each_row;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;

    // 遍历所有的列索引
    // for (unsigned long i = 0; i < col_indices_ptr->get_len(); i++)
    // {
    //     // 获取当前非零元的行索引。
    //     unsigned long cur_row_index = row_indices_ptr->read_integer_from_arr(i);
    //     if (check)
    //     {
    //         assert(cur_row_index < col_indices_of_each_row.size());
    //     }

    //     // 获取当前的列索引
    //     unsigned long cur_col_index = col_indices_ptr->read_integer_from_arr(i);
    //     col_indices_of_each_row[cur_row_index].push_back(cur_col_index);
    // }

    // 针对全局和局部的padding不一样，全局的padding单独处理
    unsigned long first_nz_of_cur_row = 0;

    if (parent_pos == GLOBAL_META)
    {
        // 查看最长的行
        unsigned long max_row_length = *max_element(row_nz_number.begin(), row_nz_number.end());

        // 查看是不是被padding了
        bool is_padding = false;
        // TODO:第二次可能可以跳过padding
        // 遍历所有的行
        for (unsigned long i = 0; i < row_num; i++)
        {
            // 当前行的非零元数量
            unsigned long cur_row_size = row_nz_number[i];
            if (check)
            {
                assert(cur_row_size <= max_row_length);
            }
            if (this->padding_with_empty_row == true)
            {
                if (max_row_length != 0)
                {
                    unsigned long target_row_length = max_row_length;

                    // 要增加的非零元数量
                    unsigned long added_row_nz_num = target_row_length - cur_row_size;

                    nnz_after_padding = nnz_after_padding + added_row_nz_num;

                    if (check)
                    {
                        // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                        {
                            cout << "modify_col_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                            assert(false);
                        }
                    }

                    // 要padding是最后一个列索引
                    for (unsigned long col_id = 0; col_id < cur_row_size; col_id++)
                    {
                        unsigned long cur_col_index = col_indices_ptr->read_integer_from_arr(col_id + first_nz_of_cur_row);
                        col_indices_of_each_row.push_back(cur_col_index);
                    }
                    first_nz_of_cur_row += cur_row_size;
                    unsigned long last_col_index = (col_indices_of_each_row.size() > 0) ? col_indices_of_each_row[col_indices_of_each_row.size() - 1] : col_indices_ptr->read_integer_from_arr(0);

                    // 增加内容
                    for (unsigned long added_nz = 0; added_nz < added_row_nz_num; added_nz++)
                    {
                        col_indices_of_each_row.push_back(last_col_index);
                    }

                    is_padding = true;
                }
            }
            else
            {
                if (max_row_length != 0 && cur_row_size != 0)
                {
                    unsigned long target_row_length = max_row_length;

                    // 要增加的非零元数量
                    unsigned long added_row_nz_num = target_row_length - cur_row_size;

                    nnz_after_padding = nnz_after_padding + added_row_nz_num;

                    if (check)
                    {
                        // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                        {
                            cout << "modify_col_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                            assert(false);
                        }
                    }
                    for (unsigned long col_id = 0; col_id < cur_row_size; col_id++)
                    {
                        unsigned long cur_col_index = col_indices_ptr->read_integer_from_arr(col_id + first_nz_of_cur_row);
                        col_indices_of_each_row.push_back(cur_col_index);
                    }
                    first_nz_of_cur_row += cur_row_size;
                    unsigned long last_col_index = (col_indices_of_each_row.size() > 0) ? col_indices_of_each_row[col_indices_of_each_row.size() - 1] : col_indices_ptr->read_integer_from_arr(0);

                    // 增加内容
                    for (unsigned long added_nz = 0; added_nz < added_row_nz_num; added_nz++)
                    {
                        col_indices_of_each_row.push_back(last_col_index);
                    }

                    is_padding = true;
                }
            }

            // 完成了padding
        }

        // 如果确实发生了padding，就记录输入，并且将记录了列索引的二维数组拉平，并且输出到metadata set
        if (is_padding == true)
        {
            // 记录当前读取的数据
            shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(row_indices_record);
            shared_ptr<data_item_record> col_indices_record(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(col_indices_record);
            shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(begin_row_index_record);
            shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(end_row_index_record);

            // 将数组拉平
            vector<unsigned long> col_indices_vec = col_indices_of_each_row;

            // for (unsigned long i = 0; i < col_indices_of_each_row.size(); i++)
            // {
            //     // 遍历每一行
            //     for (unsigned long j = 0; j < col_indices_of_each_row[i].size(); j++)
            //     {
            //         col_indices_vec.push_back(col_indices_of_each_row[i][j]);
            //     }
            // }

            if (check)
            {
                assert(col_indices_vec.size() >= row_indices_ptr->get_len());
            }

            shared_ptr<data_item_record> new_nz_col_indices_record(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
            this->dest_data_item_ptr_vec.push_back(new_nz_col_indices_record);

            // 删除对应的nz_col_indices
            this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id);

            // 将col_indices_vec转化为通用数组
            shared_ptr<universal_array> new_col_index_ptr(new universal_array(&(col_indices_vec[0]), col_indices_vec.size(), UNSIGNED_LONG));
            shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_col_index_ptr, GLOBAL_META, "nz_col_indices", this->target_matrix_id));

            this->meta_data_set_ptr->add_element(item_ptr);
        }
    }
    else if (parent_pos == TBLOCK_META || parent_pos == WARP_META)
    {
        // 读入行索引列索引，以及当前子矩阵的行边界，并且还需要父块的行偏移量索引来处理每一个子块的内容。
        // 获取父块的行偏移量索引
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
                unsigned long cur_row_size = row_nz_number[row_index];

                if (max_row_size_of_this_parent < cur_row_size)
                {
                    max_row_size_of_this_parent = cur_row_size;
                }
            }

            // 遍历当前父块的行区间，将每一行padding到对应的大小
            for (unsigned long row_index = first_row_index_of_cur_parent; row_index < first_row_index_of_next_parent; row_index++)
            {
                unsigned long cur_row_size = row_nz_number[row_index];
                if (check)
                {
                    assert(cur_row_size <= max_row_size_of_this_parent);
                }

                if (this->padding_with_empty_row == true)
                {
                    // 查看当前行是不是需要padding
                    if (max_row_size_of_this_parent != 0)
                    {
                        unsigned long target_row_length = max_row_size_of_this_parent;

                        // 要增加的非零元数量
                        unsigned long added_row_nnz = target_row_length - cur_row_size;

                        nnz_after_padding = nnz_after_padding + added_row_nnz;

                        if (check)
                        {
                            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                            {
                                cout << "modify_col_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                                assert(false);
                            }
                        }

                        for (unsigned long col_id = 0; col_id < cur_row_size; col_id++)
                        {
                            unsigned long cur_col_index = col_indices_ptr->read_integer_from_arr(col_id + first_nz_of_cur_row);
                            col_indices_of_each_row.push_back(cur_col_index);
                        }
                        first_nz_of_cur_row += cur_row_size;
                        unsigned long last_col_index = (col_indices_of_each_row.size() > 0) ? col_indices_of_each_row[col_indices_of_each_row.size() - 1] : col_indices_ptr->read_integer_from_arr(0);

                        // 增加内容
                        for (unsigned long added_nz = 0; added_nz < added_row_nnz; added_nz++)
                        {
                            col_indices_of_each_row.push_back(last_col_index);
                        }

                        is_padding = true;
                    }
                }
                else
                {
                    // 查看当前行是不是需要padding
                    if (max_row_size_of_this_parent != 0 && cur_row_size != 0)
                    {
                        unsigned long target_row_length = max_row_size_of_this_parent;

                        // 要增加的非零元数量
                        unsigned long added_row_nnz = target_row_length - cur_row_size;

                        nnz_after_padding = nnz_after_padding + added_row_nnz;

                        if (check)
                        {
                            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                            {
                                cout << "modify_col_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                                assert(false);
                            }
                        }

                        for (unsigned long col_id = 0; col_id < cur_row_size; col_id++)
                        {
                            unsigned long cur_col_index = col_indices_ptr->read_integer_from_arr(col_id + first_nz_of_cur_row);
                            col_indices_of_each_row.push_back(cur_col_index);
                        }
                        first_nz_of_cur_row += cur_row_size;
                        unsigned long last_col_index = (col_indices_of_each_row.size() > 0) ? col_indices_of_each_row[col_indices_of_each_row.size() - 1] : col_indices_ptr->read_integer_from_arr(0);

                        // 增加内容
                        for (unsigned long added_nz = 0; added_nz < added_row_nnz; added_nz++)
                        {
                            col_indices_of_each_row.push_back(last_col_index);
                        }

                        is_padding = true;
                    }
                }

                // 完成了padding
            }
        }

        // 如果存在padding就拉直二维数组，并且将列索引放到metadata set中
        if (is_padding == true)
        {
            // 记录当前读取的数据
            shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(row_indices_record);
            shared_ptr<data_item_record> col_indices_record(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(col_indices_record);
            shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(begin_row_index_record);
            shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(end_row_index_record);
            shared_ptr<data_item_record> first_row_indices_record(new data_item_record(parent_pos, "first_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(first_row_indices_record);

            // 将数组拉平
            vector<unsigned long> col_indices_vec = col_indices_of_each_row;

            // for (unsigned long i = 0; i < col_indices_of_each_row.size(); i++)
            // {
            //     // 遍历每一行
            //     for (unsigned long j = 0; j < col_indices_of_each_row[i].size(); j++)
            //     {
            //         col_indices_vec.push_back(col_indices_of_each_row[i][j]);
            //     }
            // }

            // 这个时候的列索引的长度一定大于行索引
            if (check)
            {
                assert(col_indices_vec.size() > row_indices_ptr->get_len());
            }

            shared_ptr<data_item_record> new_nz_col_indices_record(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
            this->dest_data_item_ptr_vec.push_back(new_nz_col_indices_record);

            // 删除对应的nz_col_indices
            this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id);

            // 将col_indices_vec转化为通用数组
            shared_ptr<universal_array> new_col_index_ptr(new universal_array(&(col_indices_vec[0]), col_indices_vec.size(), UNSIGNED_LONG));
            shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_col_index_ptr, GLOBAL_META, "nz_col_indices", this->target_matrix_id));

            this->meta_data_set_ptr->add_element(item_ptr);
        }
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_col_indices_by_col_pad_parent_blk_to_max_row_size::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> modify_col_indices_by_col_pad_parent_blk_to_max_row_size::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string modify_col_indices_by_col_pad_parent_blk_to_max_row_size::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    string return_str = "modify_col_indices_by_col_pad_parent_blk_to_max_row_size::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",parent_pos:" + convert_pos_type_to_string(parent_pos) + "}";

    return return_str;
}