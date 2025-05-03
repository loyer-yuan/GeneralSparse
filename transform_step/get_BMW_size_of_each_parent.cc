#include "../data_transform_step.hpp"

get_BMW_size_of_each_parent::get_BMW_size_of_each_parent(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id)
    : basic_data_transform_step("get_BMW_size_of_each_parent", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(parent_pos == GLOBAL_META || parent_pos == TBLOCK_META);

    this->parent_pos = parent_pos;
    this->target_matrix_id = target_matrix_id;
}

void get_BMW_size_of_each_parent::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->parent_pos == GLOBAL_META || this->parent_pos == TBLOCK_META);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == false);
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == false);
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == false);
        // 查看当前父块非零元偏移，和BMW的非零元偏移，都是绝对偏移，从而推测出每个父块中BMW的大小
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id));
        if (this->parent_pos != GLOBAL_META)
        {
            assert(this->meta_data_set_ptr->is_exist(this->parent_pos, "first_nz_indices", this->target_matrix_id));
        }
    }

    // 读出来数组
    shared_ptr<universal_array> first_nz_indices_of_BMW_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> first_nz_indices_of_parent_ptr = NULL;
    if (this->parent_pos != GLOBAL_META)
    {
        first_nz_indices_of_parent_ptr = this->meta_data_set_ptr->get_element(this->parent_pos, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
        // 最后一位和第一位是一样的
        if (check)
        {
            assert(first_nz_indices_of_BMW_ptr->read_integer_from_arr(0) == first_nz_indices_of_parent_ptr->read_integer_from_arr(0));
            assert(first_nz_indices_of_BMW_ptr->read_integer_from_arr(first_nz_indices_of_BMW_ptr->get_len() - 1) == first_nz_indices_of_parent_ptr->read_integer_from_arr(first_nz_indices_of_parent_ptr->get_len() - 1));
        }
    }

    // 记录输入输出
    shared_ptr<data_item_record> first_nz_indices_of_BMW_record(new data_item_record(WARP_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_of_BMW_record);
    if (this->parent_pos != GLOBAL_META)
    {
        shared_ptr<data_item_record> first_nz_indices_of_parent_record(new data_item_record(this->parent_pos, "first_nz_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(first_nz_indices_of_parent_record);
    }

    // 遍历BMW的索引，会不断自增
    unsigned long cur_BMW_id = 0;

    // 用一个数组来存储所有父块的BMW大小
    vector<unsigned long> BMW_size_of_each_parent_vec;

    if (this->parent_pos == GLOBAL_META)
    {
        // 当前的父块的BMW的大小，用来比较是不是一样的
        unsigned long cur_BMW_size_of_parent = 0;
        bool go_through_the_first_BMW = false;

        // 遍历当前父块的所有的BMW
        if (check)
        {
            assert(first_nz_indices_of_BMW_ptr->get_len() >= 2);
        }
        while (cur_BMW_id < first_nz_indices_of_BMW_ptr->get_len() - 1)
        {
            // 当前BMW的偏移量
            unsigned long cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);

            unsigned long next_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id + 1);

            if (check)
            {
                // 下一个BMW的偏移量
                if (cur_BMW_id + 1 >= first_nz_indices_of_BMW_ptr->get_len())
                {
                    cout << "get_BMW_size_of_each_parent::run(): cur_BMW_id + 1:" << cur_BMW_id + 1 << ", first_nz_indices_of_BMW_ptr->get_len():" << first_nz_indices_of_BMW_ptr->get_len() << endl;
                    assert(false);
                }
                // 当前BMW的大小
                assert(next_BMW_first_nz_id > cur_BMW_first_nz_id);
            }

            unsigned long BMW_size = next_BMW_first_nz_id - cur_BMW_first_nz_id;

            if (go_through_the_first_BMW == false)
            {
                cur_BMW_size_of_parent = BMW_size;
                go_through_the_first_BMW = true;
            }
            else
            {
                if (check)
                {
                    // 这里做一个检查，保证当前的BMW大小和之前的都是一样的
                    if (BMW_size != cur_BMW_size_of_parent)
                    {
                        cout << "get_BMW_size_of_each_parent::run(): The BMW sizes in this parent block are not the same" << endl;
                        // assert(false);
                        return;
                    }
                }
            }

            // 索引自增
            cur_BMW_id++;
            // 为了退出条件正确要查看当前的行偏移量
            cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);
        }
        if (check)
        {
            // 检查，如果当前父块不是空的，那内部的BMW的
            if (go_through_the_first_BMW == false)
            {
                assert(cur_BMW_size_of_parent == 0);
            }
            else
            {
                assert(cur_BMW_size_of_parent > 0);
            }
        }

        BMW_size_of_each_parent_vec.push_back(cur_BMW_size_of_parent);
    }
    else
    {
        // 遍历所有的父块的索引，查看内部所有的BMW的非零元索引，计算BMW的大小
        for (unsigned long parent_blk_id = 0; parent_blk_id < first_nz_indices_of_parent_ptr->get_len() - 1; parent_blk_id++)
        {
            // 当前父块和下一个父块的非零元索引
            unsigned long cur_parent_blk_first_nz_id = first_nz_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id);
            unsigned long next_parent_blk_first_nz_id = first_nz_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id + 1);
            unsigned long cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);

            if (check)
            {
                // 当前BMW的非零元偏移量
                assert(cur_BMW_id < first_nz_indices_of_BMW_ptr->get_len());

                // 在一开始的时候，BMW的非零元偏移量和父块的非零元偏移量是相同的
                // assert(cur_BMW_first_nz_id == cur_parent_blk_first_nz_id);
                if (cur_BMW_first_nz_id != cur_parent_blk_first_nz_id)
                {
                    cout << "get_BMW_size_of_each_parent::run(): cur_BMW_first_nz_id:" << cur_BMW_first_nz_id << ", cur_parent_blk_first_nz_id:" << cur_parent_blk_first_nz_id << endl;
                    assert(false);
                }
            }

            // 当前的父块的BMW的大小，用来比较是不是一样的，并且最终用来记录BMW的大小
            unsigned long cur_BMW_size_of_parent = 0;
            bool go_through_the_first_BMW = false;

            // 遍历当前父块的所有的BMW
            while (cur_BMW_first_nz_id < next_parent_blk_first_nz_id)
            {
                cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);
                unsigned long next_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id + 1);

                if (check)
                {
                    // 当前BMW的偏移量
                    assert(cur_BMW_id < first_nz_indices_of_BMW_ptr->get_len());

                    if (cur_BMW_id + 1 >= first_nz_indices_of_BMW_ptr->get_len())
                    {
                        cout << "get_BMW_size_of_each_parent::run(): cur_BMW_id + 1:" << cur_BMW_id + 1 << ", first_nz_indices_of_BMW_ptr->get_len():" << first_nz_indices_of_BMW_ptr->get_len() << endl;
                        assert(false);
                    }

                    assert(next_BMW_first_nz_id > cur_BMW_first_nz_id);
                }

                // 当前BMW的大小
                unsigned long BMW_size = next_BMW_first_nz_id - cur_BMW_first_nz_id;

                if (go_through_the_first_BMW == false)
                {
                    cur_BMW_size_of_parent = BMW_size;
                    go_through_the_first_BMW = true;
                }
                else
                {
                    if (check)
                    {
                        // 这里做一个检查，保证当前的BMW大小和之前的都是一样的
                        if (BMW_size != cur_BMW_size_of_parent)
                        {
                            cout << "get_BMW_size_of_each_parent::run(): The BMW sizes in this parent block are not the same" << endl;
                            // assert(false);
                            return;
                        }
                    }
                }

                // 索引自增
                cur_BMW_id++;

                // 为了退出条件正确要查看当前的行偏移量
                cur_BMW_first_nz_id = first_nz_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);
            }

            if (check)
            {
                // 检查，如果当前父块不是空的，那内部的BMW的
                if (go_through_the_first_BMW == false)
                {
                    assert(cur_BMW_size_of_parent == 0);
                }
                else
                {
                    assert(cur_BMW_size_of_parent > 0);
                }
            }

            BMW_size_of_each_parent_vec.push_back(cur_BMW_size_of_parent);
        }
    }

    if (check)
    {
        // 检查，BMW已经全部遍历完
        assert(cur_BMW_id == first_nz_indices_of_BMW_ptr->get_len() - 1);
        if (this->parent_pos != GLOBAL_META)
        {
            assert(BMW_size_of_each_parent_vec.size() == first_nz_indices_of_parent_ptr->get_len() - 1);
        }
    }

    // 将内容拷贝到metadata set中
    shared_ptr<universal_array> BMW_size_of_each_parent_ptr(new universal_array(&(BMW_size_of_each_parent_vec[0]), BMW_size_of_each_parent_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMW_size_of_each_parent_item(new meta_data_item(BMW_size_of_each_parent_ptr, this->parent_pos, "BMW_size_of_each_blk", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMW_size_of_each_parent_item);

    // 记录当前输出
    shared_ptr<data_item_record> BMW_size_of_each_parent_record(new data_item_record(this->parent_pos, "BMW_size_of_each_blk", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMW_size_of_each_parent_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_BMW_size_of_each_parent::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true);
    if(this->is_run == false)
    {
        vector<shared_ptr<data_item_record>> empty;
        return empty;
    }
    assert(this->parent_pos == GLOBAL_META || this->parent_pos == TBLOCK_META);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_BMW_size_of_each_parent::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true);
    assert(this->parent_pos == GLOBAL_META || this->parent_pos == TBLOCK_META);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_BMW_size_of_each_parent::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->parent_pos == GLOBAL_META || this->parent_pos == TBLOCK_META);

    string return_str = "get_BMW_size_of_each_parent::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",parent_pos:" + convert_pos_type_to_string(this->parent_pos) + "}";

    return return_str;
}