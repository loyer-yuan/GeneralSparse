#include "../data_transform_step.hpp"

modify_row_indices_by_col_pad_parent_blk_to_max_row_size::modify_row_indices_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool padding_with_empty_row)
    : basic_data_transform_step("modify_row_indices_by_col_pad_parent_blk_to_max_row_size", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    this->parent_pos = parent_pos;
    this->target_matrix_id = target_matrix_id;
    this->padding_with_empty_row = padding_with_empty_row;
}

void modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(bool check)
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

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;

    // 用一个数组来存储padding之后的行索引
    vector<unsigned long> padded_row_index;

    if (parent_pos == GLOBAL_META)
    {
        // 查看最长的行
        unsigned long max_row_length = *max_element(row_nz_number.begin(), row_nz_number.end());

        // 查看是不是被padding了
        bool is_padding = false;

        // 遍历所有的行
        for (unsigned long i = 0; i < row_nz_number.size(); i++)
        {
            // 当前行的非零元数量
            unsigned long cur_row_size = row_nz_number[i];
            if (check)
            {
                assert(cur_row_size <= max_row_length);
            }
            if (this->padding_with_empty_row == true)
            {
                // 查看是不是要执行padding
                if (max_row_length != 0)
                {
                    unsigned long target_row_length = max_row_length;

                    // 查看要增加的非零元数量
                    unsigned long added_row_nz_num = target_row_length - cur_row_size;

                    nnz_after_padding = nnz_after_padding + added_row_nz_num;

                    if (check)
                    {
                        // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                        {
                            cout << "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                            assert(false);
                        }
                    }

                    // 按照需要的行长度加入行索引
                    for (unsigned long row_nz_id = 0; row_nz_id < target_row_length; row_nz_id++)
                    {
                        padded_row_index.push_back(i);
                    }

                    if (cur_row_size != max_row_length)
                    {
                        is_padding = true;
                    }
                    if (check)
                    {
                        assert(target_row_length % max_row_length == 0);
                    }
                }
            }
            else
            {
                // 查看是不是要执行padding
                if (max_row_length != 0 && cur_row_size != 0)
                {
                    unsigned long target_row_length = max_row_length;

                    // 查看要增加的非零元数量
                    unsigned long added_row_nz_num = target_row_length - cur_row_size;

                    nnz_after_padding = nnz_after_padding + added_row_nz_num;

                    if (check)
                    {
                        // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                        {
                            cout << "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                            assert(false);
                        }
                    }

                    // 按照需要的行长度加入行索引
                    for (unsigned long row_nz_id = 0; row_nz_id < target_row_length; row_nz_id++)
                    {
                        padded_row_index.push_back(i);
                    }

                    if (cur_row_size % max_row_length != 0)
                    {
                        is_padding = true;
                    }
                    if (check)
                    {
                        assert(target_row_length % max_row_length == 0);
                    }
                }
            }
        }

        // 如果padding过了
        if (is_padding == true)
        {
            // 输入记录
            shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(row_indices_record);
            shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(begin_row_index_record);
            shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(end_row_index_record);

            // padding之后非零元数量一定变多了
            if (check)
            {
                assert(padded_row_index.size() > row_indices_ptr->get_len());
            }

            // 记录当前的输出
            shared_ptr<data_item_record> new_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->dest_data_item_ptr_vec.push_back(new_row_indices_record);

            // 删除老的数据
            this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);

            // 加入新的数据
            shared_ptr<universal_array> new_row_indices_ptr(new universal_array(&(padded_row_index[0]), padded_row_index.size(), UNSIGNED_LONG));
            shared_ptr<meta_data_item> new_row_indices_item(new meta_data_item(new_row_indices_ptr, GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->meta_data_set_ptr->add_element(new_row_indices_item);
        }
    }
    else if (parent_pos == TBLOCK_META || parent_pos == WARP_META)
    {
        // 根据父块的偏移量执行
        if (check)
        {
            assert(this->meta_data_set_ptr->is_exist(parent_pos, "first_row_indices", this->target_matrix_id));
        }
        shared_ptr<universal_array> first_row_indices_of_parent_ptr = this->meta_data_set_ptr->get_element(parent_pos, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

        // 查看是不是需要padding
        bool is_padding = false;
        if (check)
        {
            assert(padded_row_index.size() == 0);
        }
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
                if (check)
                {
                    assert(row_index < row_nz_number.size());
                }
                unsigned long cur_row_size = row_nz_number[row_index];

                // 查看当前子块的最大行长度
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
                    assert(row_index < row_nz_number.size());
                    assert(cur_row_size <= max_row_size_of_this_parent);
                }
                if (this->padding_with_empty_row == true)
                {
                    // 查看是不是要padding
                    if (max_row_size_of_this_parent != 0)
                    {
                        unsigned long target_row_length = max_row_size_of_this_parent;

                        // 增加非零元数量
                        if (check)
                        {
                            assert(target_row_length >= cur_row_size);
                        }

                        unsigned long added_row_nnz = target_row_length - cur_row_size;

                        nnz_after_padding = nnz_after_padding + added_row_nnz;
                        if (check)
                        {
                            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                            {
                                cout << "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                                assert(false);
                            }
                        }

                        // 执行padding
                        // 按照需要的行长度加入行索引
                        for (unsigned long row_nz_id = 0; row_nz_id < target_row_length; row_nz_id++)
                        {
                            padded_row_index.push_back(row_index);
                        }

                        if (cur_row_size != max_row_size_of_this_parent)
                        {
                            is_padding = true;
                        }
                        if (check)
                        {
                            assert(target_row_length % max_row_size_of_this_parent == 0);
                        }
                    }
                }
                else
                {
                    // 查看是不是要padding
                    if (max_row_size_of_this_parent != 0 && cur_row_size != 0)
                    {
                        unsigned long target_row_length = max_row_size_of_this_parent;

                        // 增加非零元数量
                        if (check)
                        {
                            assert(target_row_length >= cur_row_size);
                        }

                        unsigned long added_row_nnz = target_row_length - cur_row_size;

                        nnz_after_padding = nnz_after_padding + added_row_nnz;
                        if (check)
                        {
                            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                            {
                                cout << "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                                assert(false);
                            }
                        }

                        // 执行padding
                        // 按照需要的行长度加入行索引
                        for (unsigned long row_nz_id = 0; row_nz_id < target_row_length; row_nz_id++)
                        {
                            padded_row_index.push_back(row_index);
                        }

                        if (cur_row_size % max_row_size_of_this_parent != 0)
                        {
                            is_padding = true;
                        }
                        if (check)
                        {
                            assert(target_row_length % max_row_size_of_this_parent == 0);
                        }
                    }
                }
            }
        }

        // padding完了之后将新的数据放到metadata set中
        // 如果padding过了
        if (is_padding == true)
        {
            // 输入记录
            shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(row_indices_record);
            shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(begin_row_index_record);
            shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(end_row_index_record);
            shared_ptr<data_item_record> first_row_indices_record(new data_item_record(parent_pos, "first_row_indices", this->target_matrix_id));
            this->source_data_item_ptr_vec.push_back(first_row_indices_record);

            // padding之后非零元数量一定变多了
            if (check)
            {
                assert(padded_row_index.size() > row_indices_ptr->get_len());
            }

            // 记录当前的输出
            shared_ptr<data_item_record> new_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->dest_data_item_ptr_vec.push_back(new_row_indices_record);

            // 删除老的数据
            this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);

            // 加入新的数据
            shared_ptr<universal_array> new_row_indices_ptr(new universal_array(&(padded_row_index[0]), padded_row_index.size(), UNSIGNED_LONG));
            shared_ptr<meta_data_item> new_row_indices_item(new meta_data_item(new_row_indices_ptr, GLOBAL_META, "nz_row_indices", this->target_matrix_id));
            this->meta_data_set_ptr->add_element(new_row_indices_item);
        }
    }

    // 完成执行
    this->is_run = true;
}

// 查看当前step的输入
vector<shared_ptr<data_item_record>> modify_row_indices_by_col_pad_parent_blk_to_max_row_size::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> modify_row_indices_by_col_pad_parent_blk_to_max_row_size::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string modify_row_indices_by_col_pad_parent_blk_to_max_row_size::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    string return_str = "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",parent_pos:" + convert_pos_type_to_string(parent_pos) + "}";

    return return_str;
}