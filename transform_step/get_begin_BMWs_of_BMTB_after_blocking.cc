#include "../data_transform_step.hpp"

get_begin_BMWs_of_BMTB_after_blocking::get_begin_BMWs_of_BMTB_after_blocking(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("get_begin_BMWs_of_BMTB_after_blocking", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
}

void get_begin_BMWs_of_BMTB_after_blocking::run(bool check)
{
    if (check)
    {
        assert(meta_data_set_ptr != NULL);
        assert(meta_data_set_ptr->check());
        assert(target_matrix_id >= 0);

        // 利用父块和BMW的非零元偏移量来计算每一个父块中BMW的数量，并且使用这些数量来得到父块中BMW的偏移量
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    }

    // 读出来数组
    shared_ptr<universal_array> first_nz_indices_of_BMW_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> first_nz_indices_of_BMTB_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

    // 最后一位和第一位是一样的
    if (check)
    {
        assert(first_nz_indices_of_BMW_ptr->read_integer_from_arr(0) == first_nz_indices_of_BMTB_ptr->read_integer_from_arr(0));
        assert(first_nz_indices_of_BMW_ptr->read_integer_from_arr(first_nz_indices_of_BMW_ptr->get_len() - 1) == first_nz_indices_of_BMTB_ptr->read_integer_from_arr(first_nz_indices_of_BMTB_ptr->get_len() - 1));
    }
    // 记录输入输出
    shared_ptr<data_item_record> first_nz_indices_of_BMW_record(new data_item_record(WARP_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_of_BMW_record);
    shared_ptr<data_item_record> first_nz_indices_of_BMTB_record(new data_item_record(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_of_BMTB_record);

    // 遍历BMW的索引，会不断自增
    unsigned long cur_BMW_id = 0;

    // 用一个数组来存储所有父块的BMW大小
    vector<unsigned long> begin_BMW_index_of_BMTB_vec;
    begin_BMW_index_of_BMTB_vec.push_back(0);

    // 遍历所有的父块的索引，查看内部所有的BMW的非零元索引，计算BMW的大小
    for (unsigned long parent_blk_id = 0; parent_blk_id < first_nz_indices_of_BMTB_ptr->get_len() - 1; parent_blk_id++)
    {
        // 当前父块和下一个父块的非零元索引
        unsigned long cur_BMTB_first_nz_id = first_nz_indices_of_BMTB_ptr->read_integer_from_arr(parent_blk_id);
        unsigned long next_BMTB_first_nz_id = first_nz_indices_of_BMTB_ptr->read_integer_from_arr(parent_blk_id + 1);
        unsigned long cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);
        if (check)
        {
            // 当前BMW的非零元偏移量
            assert(cur_BMW_id < first_nz_indices_of_BMW_ptr->get_len());

            // 在一开始的时候，BMW的非零元偏移量和父块的非零元偏移量是相同的
            if (cur_BMW_first_nz_id != cur_BMTB_first_nz_id)
            {
                cout << "get_begin_BMWs_of_BMTB_after_blocking::run(): cur_BMW_first_nz_id:" << cur_BMW_first_nz_id << ", cur_parent_blk_first_nz_id:" << cur_BMTB_first_nz_id << endl;
                assert(false);
            }
        }

        // 当前父块的BMW数量
        unsigned long cur_BMW_num = 0;

        // 遍历当前父块的所有的BMW
        while (cur_BMW_first_nz_id < next_BMTB_first_nz_id)
        {
            cur_BMW_num++;

            cur_BMW_id++;
            if (check)
            {
                assert(cur_BMW_id < first_nz_indices_of_BMW_ptr->get_len());
            }
            cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);
        }
        if (check)
        {

            assert(cur_BMW_first_nz_id == next_BMTB_first_nz_id);
            // 当前BMW的数量
        }
        begin_BMW_index_of_BMTB_vec.push_back(begin_BMW_index_of_BMTB_vec[begin_BMW_index_of_BMTB_vec.size() - 1] + cur_BMW_num);
        // 当前父块中，BMW偏移，和当前遍历到的BMW块的块号相同
        if (check)
        {
            assert(begin_BMW_index_of_BMTB_vec[begin_BMW_index_of_BMTB_vec.size() - 1] == cur_BMW_id);
        }
    }

    // 检查各个数据的大小
    if (check)
    {
        assert(begin_BMW_index_of_BMTB_vec.size() == first_nz_indices_of_BMTB_ptr->get_len());
        assert(begin_BMW_index_of_BMTB_vec[begin_BMW_index_of_BMTB_vec.size() - 1] == first_nz_indices_of_BMW_ptr->get_len() - 1);
        assert(cur_BMW_id == first_nz_indices_of_BMW_ptr->get_len() - 1);
    }

    // 将内容拷贝到metadata set中
    shared_ptr<universal_array> begin_BMW_index_of_BMTB_ptr(new universal_array(&(begin_BMW_index_of_BMTB_vec[0]), begin_BMW_index_of_BMTB_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> begin_BMW_index_of_BMTB_item(new meta_data_item(begin_BMW_index_of_BMTB_ptr, TBLOCK_META, "first_BMW_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(begin_BMW_index_of_BMTB_item);

    // 记录输出记录
    shared_ptr<data_item_record> begin_BMW_index_of_BMTB_record(new data_item_record(TBLOCK_META, "first_BMW_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(begin_BMW_index_of_BMTB_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_BMWs_of_BMTB_after_blocking::get_source_data_item_ptr_in_data_transform_step()
{
    assert(target_matrix_id >= 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_BMWs_of_BMTB_after_blocking::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(target_matrix_id >= 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_BMWs_of_BMTB_after_blocking::convert_to_string()
{
    assert(target_matrix_id >= 0);
    assert(this->is_run == true);

    string return_str = "get_begin_BMWs_of_BMTB_after_blocking::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}