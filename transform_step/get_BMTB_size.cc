#include "../data_transform_step.hpp"

get_BMTB_size::get_BMTB_size(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("get_BMTB_size", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
}

void get_BMTB_size::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == false);
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == false);
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == false);

        // 查看当前父块非零元偏移，和BMTB的非零元偏移，都是绝对偏移，从而推测出每个父块中BMTB的大小
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    }

    // 读出来数组
    shared_ptr<universal_array> first_nz_indices_of_BMTB_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录输入输出
    shared_ptr<data_item_record> first_nz_indices_of_BMTB_record(new data_item_record(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(first_nz_indices_of_BMTB_record);

    // 遍历BMTB的索引，会不断自增
    unsigned long cur_BMTB_id = 0;

    // 用一个数组来存储所有父块的BMTB大小
    vector<unsigned long> BMTB_size_of_each_parent_vec;

    // 当前的父块的BMTB的大小，用来比较是不是一样的
    unsigned long cur_BMTB_size_of_parent = 0;
    bool go_through_the_first_BMTB = false;

    // 遍历当前父块的所有的BMTB
    if (check)
    {
        assert(first_nz_indices_of_BMTB_ptr->get_len() >= 2);
    }
    while (cur_BMTB_id < first_nz_indices_of_BMTB_ptr->get_len() - 1)
    {
        // 当前BMTB的偏移量
        unsigned long cur_BMTB_first_nz_id = first_nz_indices_of_BMTB_ptr->read_integer_from_arr(cur_BMTB_id);
        unsigned long next_BMTB_first_nz_id = first_nz_indices_of_BMTB_ptr->read_integer_from_arr(cur_BMTB_id + 1);

        // 下一个BMTB的偏移量
        if (check)
        {
            if (cur_BMTB_id + 1 >= first_nz_indices_of_BMTB_ptr->get_len())
            {
                cout << "get_BMTB_size_of_each_parent::run(): cur_BMTB_id + 1:" << cur_BMTB_id + 1 << ", first_nz_indices_of_BMTB_ptr->get_len():" << first_nz_indices_of_BMTB_ptr->get_len() << endl;
                assert(false);
            }
            // 当前BMTB的大小
            assert(next_BMTB_first_nz_id > cur_BMTB_first_nz_id);
        }

        unsigned long BMTB_size = next_BMTB_first_nz_id - cur_BMTB_first_nz_id;

        if (go_through_the_first_BMTB == false)
        {
            cur_BMTB_size_of_parent = BMTB_size;
            go_through_the_first_BMTB = true;
        }
        else
        {
            if (check)
            {
                // 这里做一个检查，保证当前的BMTB大小和之前的都是一样的
                if (BMTB_size != cur_BMTB_size_of_parent)
                {
                    cout << "get_BMTB_size_of_each_parent::run(): The BMTB sizes in this parent block are not the same" << endl;
                    assert(false);
                }
            }
        }

        // 索引自增
        cur_BMTB_id++;
        // 为了退出条件正确要查看当前的行偏移量
        cur_BMTB_first_nz_id = first_nz_indices_of_BMTB_ptr->read_integer_from_arr(cur_BMTB_id);
    }
    if (check)
    {
        // 检查，如果当前父块不是空的，那内部的BMTB的
        if (go_through_the_first_BMTB == false)
        {
            assert(cur_BMTB_size_of_parent == 0);
        }
        else
        {
            assert(cur_BMTB_size_of_parent > 0);
        }
    }

    BMTB_size_of_each_parent_vec.push_back(cur_BMTB_size_of_parent);

    // 检查，BMTB已经全部遍历完
    if (check)
    {
        assert(cur_BMTB_id == first_nz_indices_of_BMTB_ptr->get_len() - 1);
    }

    // 将内容拷贝到metadata set中
    shared_ptr<universal_array> BMTB_size_of_each_parent_ptr(new universal_array(&(BMTB_size_of_each_parent_vec[0]), BMTB_size_of_each_parent_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMTB_size_of_each_parent_item(new meta_data_item(BMTB_size_of_each_parent_ptr, GLOBAL_META, "BMTB_size_of_each_blk", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMTB_size_of_each_parent_item);

    // 记录当前输出
    shared_ptr<data_item_record> BMTB_size_of_each_parent_record(new data_item_record(GLOBAL_META, "BMTB_size_of_each_blk", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMTB_size_of_each_parent_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_BMTB_size::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_BMTB_size::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_BMTB_size::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "get_BMTB_size::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",parent_pos:" + "GLOBAL_META" + "}";

    return return_str;
}