#include "../data_transform_step.hpp"

get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size)
    : basic_data_transform_step("get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(col_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->col_size = col_size;
}

void get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->col_size > 0);
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

    // 获得每一行的非零元的数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);
    if (check)
    {
        assert(end_row_index >= start_row_index);
        // 查看当前上一个父块的类型
        bool BMTB_blocking_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
        bool BMW_blocking_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);

        // 父切块必须是行切块
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

            //     // 如果多次重复，代表出错
            //     if (repeat_num == 2)
            //     {
            //         cout << "get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::run(): two BMTBs have mapped to one row" << endl;
            //         assert(false);
            //     }

            //     // 记录之前的行索引
            //     prev_BMTB_first_row_index = BMTB_first_row_index_ptr->read_integer_from_arr(i);
            // }
        }

        if (BMW_blocking_existing == true)
        {
            assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, WARP_META, this->target_matrix_id));
            // BMW是父块，BMW之前肯定是行切块，做一个检查，其行偏移量索引中不可能出现三个相同的行偏移量，因为这意味着，两个父块在同一行中
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
            //         cout << "get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::run(): two BMWs have mapped to one row" << endl;
            //         assert(false);
            //     }

            //     // 记录之前的行索引
            //     prev_BMW_first_row_index = BMW_first_row_index_ptr->read_integer_from_arr(i);
            // }
        }
        assert(nnz_of_each_row.size() == row_num);
    }

    // 用一个数组存储每一个BMT的非零元偏移量
    vector<unsigned long> BMT_begin_nz_vec;
    BMT_begin_nz_vec.push_back(0);

    // 遍历每一行，查看对应的BMT的非零元偏移量，BMT原则上不存在空的
    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        // 查看当前行非零元的数量
        long cur_row_size = nnz_of_each_row[i];

        // 剩下的行长度，用来处理最后一个偏移量
        long remain_row_size = cur_row_size;

        while (remain_row_size > 0)
        {
            // 看看能不能插入完整大小的BMT
            if (remain_row_size < this->col_size)
            {
                BMT_begin_nz_vec.push_back(BMT_begin_nz_vec[BMT_begin_nz_vec.size() - 1] + remain_row_size);
                remain_row_size = remain_row_size - remain_row_size;
            }
            else
            {
                BMT_begin_nz_vec.push_back(BMT_begin_nz_vec[BMT_begin_nz_vec.size() - 1] + this->col_size);
                remain_row_size = remain_row_size - this->col_size;
            }
        }
        if (check)
        {
            assert(remain_row_size == 0);
        }
    }
    if (check)
    {
        assert(nz_row_indices_ptr->get_len() == BMT_begin_nz_vec[BMT_begin_nz_vec.size() - 1]);
    }
    // 到这里，BMT_begin_nz_vec的最后一位是总非零元数量

    // 将新的内容放到metadata set中
    shared_ptr<universal_array> BMT_begin_nz_ptr(new universal_array(&(BMT_begin_nz_vec[0]), BMT_begin_nz_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMT_begin_nz_item(new meta_data_item(BMT_begin_nz_ptr, THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMT_begin_nz_item);

    // 加入新的record
    shared_ptr<data_item_record> BMT_begin_nz_record(new data_item_record(THREAD_META, "first_nz_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMT_begin_nz_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->col_size > 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->col_size > 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->col_size > 0);

    string return_str = "get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",col_size:" + to_string(this->col_size) + "}";

    return return_str;
}