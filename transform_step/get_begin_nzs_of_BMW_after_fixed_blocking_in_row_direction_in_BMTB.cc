#include "data_transform_step.hpp"

get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size)
    : basic_data_transform_step("get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(fixed_row_block_size > 0);
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_block_size = fixed_row_block_size;
}

void get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->fixed_row_block_size >= 1);
        // 检查
        assert(this->meta_data_set_ptr->check());

        // 存在TBLOCK级别的数据
        assert(this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id) > 0);

        // 需要当前子块的边界行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 查看有没有行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // tblock的行边界
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id));
        // tblock的值边界
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    }

    // 读出来当前子块行边界
    unsigned long start_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 当前的行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    // tblock行边界
    shared_ptr<universal_array> BMTB_first_row_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
    // tblock值边界
    shared_ptr<universal_array> BMTB_first_nzs_ptr = NULL;
    if (check)
    {
        BMTB_first_nzs_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
        assert(BMTB_first_nzs_ptr->get_len() == BMTB_first_row_indices_ptr->get_len());

        assert(end_row_index >= start_row_index);
    }

    // 真正的行结束边界
    unsigned long real_end_row_index = start_row_index + nz_row_indices_ptr->read_integer_from_arr(nz_row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }

    // 当前的行数量
    unsigned long row_num = end_row_index - start_row_index + 1;

    // 每一行非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= start_row_index);
        assert(nnz_of_each_row.size() == row_num);
    }

    vector<unsigned long> nz_begin_index_of_each_BMW;
    nz_begin_index_of_each_BMW.push_back(0);

    // 遍历每一个BMTB块
    for (unsigned long i = 0; i < BMTB_first_row_indices_ptr->get_len() - 1; i++)
    {
        // 当前BMTB行号
        unsigned long BMTB_first_row = BMTB_first_row_indices_ptr->read_integer_from_arr(i);
        unsigned long next_BMTB_first_row = BMTB_first_row_indices_ptr->read_integer_from_arr(i + 1);

        if (check)
        {
            unsigned long BMTB_first_nz = BMTB_first_nzs_ptr->read_integer_from_arr(i);
            // BMTB块的起始索引，是之前BMW块的叠加结果
            if (nz_begin_index_of_each_BMW[nz_begin_index_of_each_BMW.size() - 1] != BMTB_first_nz)
            {
                cout << "get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::run(): nz_begin_index_of_each_BMW[nz_begin_index_of_each_BMW.size() - 1]=" << nz_begin_index_of_each_BMW[nz_begin_index_of_each_BMW.size() - 1] << ",BMTB_first_nz=" << BMTB_first_nz << endl;
                cout << "i=" << i << endl;
                assert(nz_begin_index_of_each_BMW[nz_begin_index_of_each_BMW.size() - 1] == BMTB_first_nz);
            }
        }

        // 遍历所有BMW
        for (unsigned long BMW_first_row_index = BMTB_first_row; BMW_first_row_index < next_BMTB_first_row; BMW_first_row_index = BMW_first_row_index + this->fixed_row_block_size)
        {
            // 用一个变量存储当前BMW的非零元数量
            unsigned long BMW_nz_number = 0;

            // 遍历当前BMW的所有行
            for (unsigned long row_index = BMW_first_row_index; row_index < next_BMTB_first_row && row_index < BMW_first_row_index + this->fixed_row_block_size; row_index++)
            {
                // 叠加行数量
                if (check)
                {
                    assert(row_index <= end_row_index);
                }
                BMW_nz_number = BMW_nz_number + nnz_of_each_row[row_index];
            }

            nz_begin_index_of_each_BMW.push_back(BMW_nz_number + nz_begin_index_of_each_BMW[nz_begin_index_of_each_BMW.size() - 1]);
        }
    }
    if (check)
    {
        // 将绝对的warp索引写出来
        assert(nz_begin_index_of_each_BMW[nz_begin_index_of_each_BMW.size() - 1] == nz_row_indices_ptr->get_len());
        assert(nz_begin_index_of_each_BMW.size() >= 2);
    }

    // 将内容写到metadata set中
    shared_ptr<universal_array> BMW_begin_nz_index_ptr(new universal_array(&(nz_begin_index_of_each_BMW[0]), nz_begin_index_of_each_BMW.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> item_ptr(new meta_data_item(BMW_begin_nz_index_ptr, WARP_META, "first_nz_indices", this->target_matrix_id));

    // 加入元素
    this->meta_data_set_ptr->add_element(item_ptr);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    vector<shared_ptr<data_item_record>> record_vec;

    // 两个行边界
    shared_ptr<data_item_record> row_end_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    record_vec.push_back(row_end_record);
    shared_ptr<data_item_record> row_start_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    record_vec.push_back(row_start_record);

    // 一个行索引
    shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    record_vec.push_back(row_indices_record);

    // 线程块的行偏移量
    shared_ptr<data_item_record> BMTB_row_indices_record(new data_item_record(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    record_vec.push_back(BMTB_row_indices_record);

    // shared_ptr<data_item_record> BMTB_nz_indices_record(new data_item_record(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    // record_vec.push_back(BMTB_nz_indices_record);

    return record_vec;
}

// 获得输出节点
vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size >= 1);

    vector<shared_ptr<data_item_record>> record_vec;

    shared_ptr<data_item_record> first_nz_indices_record(new data_item_record(WARP_META, "first_nz_indices", this->target_matrix_id));
    record_vec.push_back(first_nz_indices_record);

    return record_vec;
}

string get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->fixed_row_block_size > 0);

    string return_str = "get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}