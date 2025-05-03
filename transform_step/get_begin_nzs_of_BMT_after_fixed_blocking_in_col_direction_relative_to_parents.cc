#include "../data_transform_step.hpp"

get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents::get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size, POS_TYPE parent_pos)
    : basic_data_transform_step("get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(col_size > 0);
    assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);

    this->target_matrix_id = target_matrix_id;
    this->col_size = col_size;
    this->parent_pos = parent_pos;
}

void get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->col_size > 0);
        // 检查
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);

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

    // 查看真正的行结束位置，如果有padding，real_end_row_index可能会更大
    unsigned long real_end_row_index = start_row_index + nz_row_indices_ptr->read_integer_from_arr(nz_row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }
    unsigned long row_num = end_row_index - start_row_index + 1;


    // 查看当前上一个父块的类型
    if (check)
    {        
        assert(end_row_index >= start_row_index);
        bool BMTB_blocking_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
        bool BMW_blocking_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);

        // 父块必须存在
        assert(BMTB_blocking_existing == true || BMW_blocking_existing == true);

        // 父块必须是行切分
        if (BMTB_blocking_existing == true)
        {
            assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
        }

        if (BMW_blocking_existing == true)
        {
            assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, WARP_META, this->target_matrix_id));
        }
    }

    // 获取每一行的非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(nnz_of_each_row.size() == row_num);

        // 获取当前父块的非零元索引和行索引
        assert(this->meta_data_set_ptr->is_exist(this->parent_pos, "first_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(this->parent_pos, "first_nz_indices", this->target_matrix_id));
    }
    shared_ptr<universal_array> parent_first_row_indices_ptr = this->meta_data_set_ptr->get_element(this->parent_pos, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> parent_first_nz_indices_ptr = this->meta_data_set_ptr->get_element(this->parent_pos, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

    if (check)
    {
        assert(parent_first_row_indices_ptr->read_integer_from_arr(0) == 0);
        assert(parent_first_row_indices_ptr->read_integer_from_arr(parent_first_row_indices_ptr->get_len() - 1) == row_num);
        assert(parent_first_nz_indices_ptr->read_integer_from_arr(0) == 0);
        assert(parent_first_nz_indices_ptr->read_integer_from_arr(parent_first_nz_indices_ptr->get_len() - 1) == nz_row_indices_ptr->get_len());
        assert(parent_first_row_indices_ptr->get_len() == parent_first_nz_indices_ptr->get_len());
    }

    // 记录读入的数据
    shared_ptr<data_item_record> parent_first_row_indices_record(new data_item_record(this->parent_pos, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(parent_first_row_indices_record);
    // shared_ptr<data_item_record> parent_first_nz_indices_record(new data_item_record(this->parent_pos, "first_nz_indices", this->target_matrix_id));
    // this->source_data_item_ptr_vec.push_back(parent_first_nz_indices_record);

    // 用一个数组存储所有的BMT的非零元相对偏移
    vector<unsigned long> BMT_nz_offset;

    // 遍历所有的父块，父块数量为父块的行索引与非零元索引长度-1
    for (unsigned int parent_blk = 0; parent_blk < parent_first_row_indices_ptr->get_len() - 1; parent_blk++)
    {
        // 获取当前快的行号和非零元索引
        unsigned long first_row_index_of_this_parent_blk = parent_first_row_indices_ptr->read_integer_from_arr(parent_blk);
        unsigned long first_row_index_of_next_parent_blk = parent_first_row_indices_ptr->read_integer_from_arr(parent_blk + 1);
        unsigned long first_nz_index_of_this_parent_blk = parent_first_nz_indices_ptr->read_integer_from_arr(parent_blk);
        unsigned long first_nz_index_of_next_parent_blk = parent_first_nz_indices_ptr->read_integer_from_arr(parent_blk + 1);

        if (check)
        {
            assert(first_row_index_of_this_parent_blk < first_row_index_of_next_parent_blk);
        }

        // 当前父块的第一个BMT相对nz偏移是从0开始的
        BMT_nz_offset.push_back(0);

        // 遍历当前父块的所有行
        for (unsigned int row_of_parent_blk = first_row_index_of_this_parent_blk; row_of_parent_blk < first_row_index_of_next_parent_blk; row_of_parent_blk++)
        {
            // 当前行的非零元数量
            if (check)
            {
                assert(row_of_parent_blk < row_num);
            }
            long row_length = nnz_of_each_row[row_of_parent_blk];

            long remain_row_nz_num = row_length;

            // 一直减到没
            while (remain_row_nz_num > 0)
            {
                long added_nz_num = this->col_size;

                // 如果要加上的偏移量比剩余的行非零元数量还大，那就按照剩余的行非零元数量来算
                if (added_nz_num > remain_row_nz_num)
                {
                    added_nz_num = remain_row_nz_num;
                }

                // 增加一个偏移量
                BMT_nz_offset.push_back(BMT_nz_offset[BMT_nz_offset.size() - 1] + added_nz_num);

                remain_row_nz_num = remain_row_nz_num - added_nz_num;
            }
            if (check)
            {
                assert(remain_row_nz_num == 0);
            }
        }

        // 最后一个BMT的相对偏移加上当前父块的绝对nz偏移量是下一个父块的nz偏移
        if (check)
        {
            assert(first_nz_index_of_this_parent_blk + BMT_nz_offset[BMT_nz_offset.size() - 1] == first_nz_index_of_next_parent_blk);
        }

        // 最后一个元素删掉，相对偏移没有CSR-like的尾巴，如果父块中是空的，那就没有BMT的相对索引被插入
        unsigned long cur_BMT_num = BMT_nz_offset.size();
        BMT_nz_offset.pop_back();
        // 检查一下是不是删了
        if (check)
        {
            assert(BMT_nz_offset.size() == cur_BMT_num - 1);
        }
    }

    // 当前的BMT非零元相对偏移存到metadata set中，在插入之前肯定是不存在的
    string metadata_item_name = "first_nz_indices_relative_to_";

    if (this->parent_pos == TBLOCK_META)
    {
        metadata_item_name = metadata_item_name + "BMTB";
    }
    else if (this->parent_pos == WARP_META)
    {
        metadata_item_name = metadata_item_name + "BMW";
    }
    else
    {
        assert(false);
    }
    if (check)
    {
        assert(!this->meta_data_set_ptr->is_exist(THREAD_META, metadata_item_name, this->target_matrix_id));
    }

    // 创建一个新的通用数组
    shared_ptr<universal_array> first_nz_indices_relative_to_parents_ptr(new universal_array(&(BMT_nz_offset[0]), BMT_nz_offset.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> first_nz_indices_relative_to_parents_item(new meta_data_item(first_nz_indices_relative_to_parents_ptr, THREAD_META, metadata_item_name, this->target_matrix_id));
    // 将内容放到metadata set中
    this->meta_data_set_ptr->add_element(first_nz_indices_relative_to_parents_item);

    // 增加一个输出的记录
    shared_ptr<data_item_record> first_nz_indices_relative_to_parents_record(new data_item_record(THREAD_META, metadata_item_name, this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(first_nz_indices_relative_to_parents_record);

    // 执行完毕
    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents::get_dest_data_item_ptr_in_data_transform_step_without_check()
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

string get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->col_size > 0);

    string return_str = "get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",col_size:" + to_string(this->col_size) + ",parent_pos:" + convert_pos_type_to_string(this->parent_pos) + "}";

    return return_str;
}