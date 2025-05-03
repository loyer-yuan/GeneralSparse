#include "../data_transform_step.hpp"

modify_row_indices_by_interlance_storage::modify_row_indices_by_interlance_storage(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id)
    : basic_data_transform_step("modify_row_indices_by_interlance_storage", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    this->meta_data_set_ptr = meta_data_set_ptr;
    this->parent_pos = parent_pos;
    this->target_matrix_id = target_matrix_id;
}

void modify_row_indices_by_interlance_storage::run(bool check)
{
    if (check)
    {
        assert(this->target_matrix_id >= 0);
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->parent_pos == GLOBAL_META || this->parent_pos == TBLOCK_META || this->parent_pos == WARP_META);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == false);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        if (this->parent_pos != GLOBAL_META)
        {
            assert(this->meta_data_set_ptr->is_exist(this->parent_pos, "first_BMT_indices", this->target_matrix_id));
        }
        assert(this->meta_data_set_ptr->is_exist(this->parent_pos, "BMT_size_of_each_blk", this->target_matrix_id));
    }

    // 列索引
    shared_ptr<universal_array> row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // BMT size
    shared_ptr<universal_array> BMT_size = this->meta_data_set_ptr->get_element(this->parent_pos, "BMT_size_of_each_blk", this->target_matrix_id)->get_metadata_arr();

    // 新索引
    vector<unsigned long> new_row_indices(row_indices_ptr->get_len());

    // 针对全局和局部的不一样，全局的单独处理
    if (parent_pos == GLOBAL_META)
    {
        //输入
        shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(row_indices_record);
        shared_ptr<data_item_record> BMT_size_record(new data_item_record(GLOBAL_META, "BMT_size_of_each_blk", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(BMT_size_record);
        if (check)
        {
            //检查是否是BMT_size的倍数
            assert(row_indices_ptr->get_len() % BMT_size->read_integer_from_arr(0) == 0);
        }

        //交错存储间隔
        unsigned long size_of_BMT = BMT_size->read_integer_from_arr(0);
        unsigned long BMT_num = row_indices_ptr->get_len() / size_of_BMT;
        unsigned long spacing = BMT_num;

        //交错存储
        for (int BMT_id = 0; BMT_id < BMT_num; BMT_id++)
        {
            for (int nz_in_BMT = 0; nz_in_BMT < size_of_BMT; nz_in_BMT++)
            {
                new_row_indices[BMT_id + nz_in_BMT * spacing] = row_indices_ptr->read_integer_from_arr(nz_in_BMT + BMT_id * size_of_BMT);
            }
        }
    }
    else if (parent_pos == TBLOCK_META || parent_pos == WARP_META)
    {
        //输入
        shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(row_indices_record);
        shared_ptr<data_item_record> BMT_size_record(new data_item_record(this->parent_pos, "BMT_size_of_each_blk", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(BMT_size_record);
        shared_ptr<data_item_record> first_BMT_indices_record(new data_item_record(this->parent_pos, "first_BMT_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(first_BMT_indices_record);

        //获取每个块的BMT偏移
        shared_ptr<universal_array> first_BMT_indices_of_parent_ptr = this->meta_data_set_ptr->get_element(this->parent_pos, "first_BMT_indices", this->target_matrix_id)->get_metadata_arr();

        unsigned long temp_size = 0;

        //遍历所有parent
        for (int parent_blk_id = 0; parent_blk_id < first_BMT_indices_of_parent_ptr->get_len() - 1; parent_blk_id++)
        {
            // 当前父块和下一个父块的非零元索引
            unsigned long cur_parent_blk_first_BMT_id = first_BMT_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id);
            unsigned long next_parent_blk_first_BMT_id = first_BMT_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id + 1);

            //父块内BMT个数
            unsigned long BMT_num = next_parent_blk_first_BMT_id - cur_parent_blk_first_BMT_id;

            // BMT的size
            unsigned long size_of_BMT = BMT_size->read_integer_from_arr(parent_blk_id);

            //交错存储间隔
            unsigned long spacing = BMT_num;

            //交错存储
            for (int BMT_id = 0; BMT_id < BMT_num; BMT_id++)
            {
                for (int nz_in_BMT = 0; nz_in_BMT < size_of_BMT; nz_in_BMT++)
                {
                    new_row_indices[BMT_id + nz_in_BMT * spacing + temp_size] = row_indices_ptr->read_integer_from_arr(nz_in_BMT + BMT_id * size_of_BMT + temp_size);
                }
            }
            if (check)
            {
                shared_ptr<universal_array> first_nz_indices_of_parent_ptr = this->meta_data_set_ptr->get_element(this->parent_pos, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
                assert(first_nz_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id) == temp_size);
            }
            temp_size += BMT_num * size_of_BMT;
        }
    }

    //输出
    shared_ptr<data_item_record> new_nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(new_nz_row_indices_record);

    // 将new_row_indices转化为通用数组
    shared_ptr<universal_array> new_row_index_ptr(new universal_array(&(new_row_indices[0]), new_row_indices.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> item_ptr(new meta_data_item(new_row_index_ptr, GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id));

    this->meta_data_set_ptr->add_element(item_ptr);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_row_indices_by_interlance_storage::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    // 空指针检查
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_row_indices_by_interlance_storage::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_row_indices_by_interlance_storage::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->parent_pos == GLOBAL_META || parent_pos == TBLOCK_META || parent_pos == WARP_META);

    string return_str = "modify_row_indices_by_interlance_storage::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",parent_pos:" + convert_pos_type_to_string(this->parent_pos) + "}";

    return return_str;
}