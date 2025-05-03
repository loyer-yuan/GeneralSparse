#include "../data_transform_step.hpp"

get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size)
    : basic_data_transform_step("get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(col_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->col_size = col_size;
}

void get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->col_size > 0);
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
    // 获得每一行的行长度

    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= start_row_index);

        // 查看当前上一个父块的类型
        bool BMTB_blocking_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
        bool BMW_blocking_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);

        assert(BMW_blocking_existing == true);

        // 之前的父块只有行切块
        if (BMTB_blocking_existing == true)
        {
            assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
            // // BMTB为父块，BMTB之前肯定是行切块，做检查
            // shared_ptr<universal_array> BMTB_first_row_index_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

            // // 用一个变量存储上一个BMTB首行索引
            // unsigned long prev_BMTB_first_row_index = BMTB_first_row_index_ptr->read_integer_from_arr(0);
            // // 查看重复的次数
            // unsigned long repeat_num = 1;

            // for (unsigned long i = 1; i < BMTB_first_row_index_ptr->get_len(); i++)
            // {
            //     if (BMTB_first_row_index_ptr->read_integer_from_arr(i) == prev_BMTB_first_row_index)
            //     {
            //         repeat_num++;
            //     }
            //     else
            //     {
            //         repeat_num = 1;
            //     }

            //     // 如果出现了两次，那就直接退出，并且说明不合法
            //     if (repeat_num == 2)
            //     {
            //         cout << "get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::run(): two BMTBs have mapped to one row" << endl;
            //         assert(false);
            //     }

            //     // 记录之前的行索引
            //     prev_BMTB_first_row_index = BMTB_first_row_index_ptr->read_integer_from_arr(i);
            // }
        }

        if (BMW_blocking_existing == true)
        {
            assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, WARP_META, this->target_matrix_id));
            // // BMW是父块，BMW之前肯定是行切块，做一个检查，其行偏移量索引中不可能出现三个相同的行偏移量，因为这意味着，两个父块在同一行中
            // shared_ptr<universal_array> BMW_first_row_index_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

            // // 遍历所有内容，用一个变量存储上一个首行索引
            // unsigned long prev_BMW_first_row_index = BMW_first_row_index_ptr->read_integer_from_arr(0);
            // // 查看重复的次数
            // unsigned long repeat_num = 1;

            // for (unsigned long i = 1; i < BMW_first_row_index_ptr->get_len(); i++)
            // {
            //     // 如果和之前的一样，就记录重复的数量
            //     if (BMW_first_row_index_ptr->read_integer_from_arr(i) == prev_BMW_first_row_index)
            //     {
            //         repeat_num++;
            //     }
            //     else
            //     {
            //         repeat_num = 1;
            //     }

            //     // 如果出现了两次，那就直接退出，并且说明不合法
            //     if (repeat_num == 2)
            //     {
            //         cout << "get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::run(): two BMWs have mapped to one row" << endl;
            //         assert(false);
            //     }

            //     // 记录之前的行索引
            //     prev_BMW_first_row_index = BMW_first_row_index_ptr->read_integer_from_arr(i);
            // }
        }
        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> BMT_begin_row_vec_relative_to_BMW;

    // 取出所有的BMTB行偏移量
    shared_ptr<universal_array> BMW_first_row_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录取出的BMTB行偏移
    shared_ptr<data_item_record> BMW_first_row_indices_record(new data_item_record(WARP_META, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(BMW_first_row_indices_record);

    // 遍历所有的BMW
    for (unsigned long i = 0; i < BMW_first_row_indices_ptr->get_len() - 1; i++)
    {
        // BMW的行边界
        unsigned long BMW_first_row_index = BMW_first_row_indices_ptr->read_integer_from_arr(i);
        unsigned long next_BMW_first_row_index = BMW_first_row_indices_ptr->read_integer_from_arr(i + 1);
        if (check)
        {
            assert(BMW_first_row_index < next_BMW_first_row_index);
        }

        // 遍历所有的行边界
        for (unsigned long j = BMW_first_row_index; j < next_BMW_first_row_index; j++)
        {
            // 查看对应的行的长度
            unsigned long cur_row_size = nnz_of_each_row[j];

            // 当前行的BMT数量
            unsigned long BMT_num_of_row = cur_row_size / this->col_size;

            // 如果不能整除，那就加一个BMT
            if (cur_row_size % this->col_size != 0)
            {
                BMT_num_of_row = BMT_num_of_row + 1;
            }

            // 遍历所有的BMT，写每一个BMT的首行索引
            for (unsigned long BMT_id_of_row = 0; BMT_id_of_row < BMT_num_of_row; BMT_id_of_row++)
            {
                // 相对索引
                BMT_begin_row_vec_relative_to_BMW.push_back(j - BMW_first_row_index);
            }
        }
    }
    if (check)
    {
        assert(BMT_begin_row_vec_relative_to_BMW.size() > 0);
    }

    // 将相对索引放到metadata set中
    shared_ptr<universal_array> BMT_first_row_index_relative_to_BMW_ptr(new universal_array(&(BMT_begin_row_vec_relative_to_BMW[0]), BMT_begin_row_vec_relative_to_BMW.size(), UNSIGNED_LONG));
    // 行偏移的相对索引
    shared_ptr<meta_data_item> BMT_first_row_index_relative_to_BMW_item(new meta_data_item(BMT_first_row_index_relative_to_BMW_ptr, THREAD_META, "first_row_indices_relative_to_BMW", this->target_matrix_id));
    // 将新的内容写到metadata set中
    this->meta_data_set_ptr->add_element(BMT_first_row_index_relative_to_BMW_item);

    // 执行记录
    shared_ptr<data_item_record> BMT_first_row_index_relative_to_BMW_record(new data_item_record(THREAD_META, "first_row_indices_relative_to_BMW", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMT_first_row_index_relative_to_BMW_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::get_source_data_item_ptr_in_data_transform_step()
{
    // 已经执行完成，并且参数满足要求
    assert(this->is_run == true);
    assert(this->col_size > 0);
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // 内容满足要求
    assert(target_matrix_id >= 0);
    assert(col_size > 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::convert_to_string()
{
    assert(target_matrix_id >= 0);
    assert(col_size > 0);

    string return_str = "get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",col_size:" + to_string(this->col_size) + "}";

    return return_str;
}