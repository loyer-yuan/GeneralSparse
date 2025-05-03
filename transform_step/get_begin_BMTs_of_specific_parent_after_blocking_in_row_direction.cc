#include "../data_transform_step.hpp"

get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id)
    : basic_data_transform_step("get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);

    this->parent_pos = parent_pos;
    this->target_matrix_id = target_matrix_id;
}

void get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::run(bool check)
{
    if (check)
    {
        assert(meta_data_set_ptr != NULL);
        assert(meta_data_set_ptr->check());
        assert(target_matrix_id >= 0);
        assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);

        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(parent_pos, "first_row_indices", this->target_matrix_id));
    }

    // 读出来数组
    shared_ptr<universal_array> first_row_indices_of_BMT_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> first_row_indices_of_parent_ptr = this->meta_data_set_ptr->get_element(parent_pos, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录输入
    shared_ptr<data_item_record> first_row_indices_of_BMT_record(new data_item_record(THREAD_META, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_row_indices_of_BMT_record);
    shared_ptr<data_item_record> first_row_indices_of_parent_record(new data_item_record(parent_pos, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_row_indices_of_parent_record);

    // 遍历BMT的索引，会不断自增
    unsigned long cur_BMT_id = 0;

    // 用一个数组来存储所有父块的BMT
    vector<unsigned long> begin_BMT_index_of_each_parent_vec;

    begin_BMT_index_of_each_parent_vec.push_back(0);

    // 遍历所有的父块的索引，查看内部所有的BMT的非零元索引，计算BMT的大小
    for (unsigned long parent_blk_id = 0; parent_blk_id < first_row_indices_of_parent_ptr->get_len() - 1; parent_blk_id++)
    {
        // 当前父块和下一个父块的非零元索引
        unsigned long cur_parent_blk_first_row_id = first_row_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id);
        unsigned long next_parent_blk_first_row_id = first_row_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id + 1);
        // 当前BMT的非零元偏移量
        unsigned long cur_BMT_first_row_id = first_row_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);

        if (check)
        {
            assert(cur_BMT_id <= first_row_indices_of_BMT_ptr->get_len());

            // 在一开始的时候，BMT的非零元偏移量和父块的非零元偏移量是相同的
            // assert(cur_BMT_first_nz_id == cur_parent_blk_first_nz_id);
            if (cur_BMT_first_row_id != cur_parent_blk_first_row_id)
            {
                cout << "get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::run(): cur_BMT_first_row_id:" << cur_BMT_first_row_id << ", cur_parent_blk_first_nz_id:" << cur_parent_blk_first_row_id << endl;
                assert(false);
            }
        }

        // 当前父块的BMT数量
        unsigned long cur_BMT_num = 0;

        // 遍历当前父块的所有的BMT
        while (cur_BMT_first_row_id < next_parent_blk_first_row_id)
        {
            cur_BMT_num++;
            cur_BMT_id++;
            if (cur_BMT_id == first_row_indices_of_BMT_ptr->get_len())
            {
                break;
            }
            cur_BMT_first_row_id = first_row_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);
        }

        // 当前BMT的总量
        begin_BMT_index_of_each_parent_vec.push_back(begin_BMT_index_of_each_parent_vec[begin_BMT_index_of_each_parent_vec.size() - 1] + cur_BMT_num);
        // 当前父块中，BMT偏移，和当前遍历到的BMT块的块号相同
        if (check)
        {
            assert(begin_BMT_index_of_each_parent_vec[begin_BMT_index_of_each_parent_vec.size() - 1] == cur_BMT_id);
        }
    }
    if (check)
    {
        // 检查各个数据的大小
        assert(begin_BMT_index_of_each_parent_vec.size() == first_row_indices_of_parent_ptr->get_len());
        assert(begin_BMT_index_of_each_parent_vec[begin_BMT_index_of_each_parent_vec.size() - 1] == first_row_indices_of_BMT_ptr->get_len() - 1);
        assert(cur_BMT_id == first_row_indices_of_BMT_ptr->get_len() - 1);
    }

    // 将内容拷贝到metadata set中
    shared_ptr<universal_array> begin_BMT_index_of_each_parent_ptr(new universal_array(&(begin_BMT_index_of_each_parent_vec[0]), begin_BMT_index_of_each_parent_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> begin_BMT_index_of_each_parent_item(new meta_data_item(begin_BMT_index_of_each_parent_ptr, parent_pos, "first_BMT_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(begin_BMT_index_of_each_parent_item);

    // 记录输出记录
    shared_ptr<data_item_record> begin_BMT_index_of_each_parent_record(new data_item_record(parent_pos, "first_BMT_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(begin_BMT_index_of_each_parent_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(target_matrix_id >= 0);
    assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(target_matrix_id >= 0);
    assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::convert_to_string()
{
    assert(target_matrix_id >= 0);
    assert(parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    string return_str = "get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",parent_pos:" + convert_pos_type_to_string(this->parent_pos) + "}";

    return return_str;
}