#include "../data_transform_step.hpp"

get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::get_begin_BMWs_of_BMTB_after_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("get_begin_BMWs_of_BMTB_after_blocking_in_row_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
}

void get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        // 检查
        assert(this->meta_data_set_ptr->check());

        // 存在TBLOCk级别的数据
        assert(this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id) > 0);

        // tblock的行边界
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    }

    shared_ptr<universal_array> BMTB_first_row_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> first_row_indices_of_BMW_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

    if (check)
    {
        assert(BMTB_first_row_indices_ptr->get_len() > 0);

        // 检查之前必须得有TBLOCK的行切分
        assert(has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
    }

    // 用一个变量来存储每个BMTB的BMW的偏移量
    vector<unsigned long> BMTB_first_BMW_indices_vec;
    BMTB_first_BMW_indices_vec.push_back(0);
    unsigned long cur_BMW_id = 0;

    // 遍历所有的BMTB
    for (unsigned long i = 0; i < BMTB_first_row_indices_ptr->get_len() - 1; i++)
    {
        // 当前BMTB
        unsigned long BMTB_first_row_index = BMTB_first_row_indices_ptr->read_integer_from_arr(i);
        unsigned long next_BMTB_first_row_index = BMTB_first_row_indices_ptr->read_integer_from_arr(i + 1);
        unsigned long cur_BMW_first_row_id = first_row_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);

        if (check)
        {
            assert(next_BMTB_first_row_index - BMTB_first_row_index >= 0);
            assert(cur_BMW_id <= first_row_indices_of_BMW_ptr->get_len());
            if (cur_BMW_first_row_id != BMTB_first_row_index)
            {
                cout << "get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::run(): cur_BMW_first_row_id:" << cur_BMW_first_row_id << ", cur_BMTB_first_row_id:" << BMTB_first_row_index << endl;
                assert(false);
            }
        }

        unsigned long cur_BMW_num = 0;
        while (cur_BMW_first_row_id < next_BMTB_first_row_index)
        {
            cur_BMW_num++;
            cur_BMW_id++;
            if (cur_BMW_id == first_row_indices_of_BMW_ptr->get_len())
            {
                break;
            }
            cur_BMW_first_row_id = first_row_indices_of_BMW_ptr->read_integer_from_arr(cur_BMW_id);
        }

        // 当前BMW的数量
        BMTB_first_BMW_indices_vec.push_back(BMTB_first_BMW_indices_vec[BMTB_first_BMW_indices_vec.size() - 1] + cur_BMW_num);
        // 当前父块中，BMT偏移，和当前遍历到的BMT块的块号相同
        if (check)
        {
            assert(BMTB_first_BMW_indices_vec[BMTB_first_BMW_indices_vec.size() - 1] == cur_BMW_id);
        }

    }

    // 将数组放到metadata set中
    shared_ptr<universal_array> BMTB_first_BMW_indices_ptr(new universal_array(&(BMTB_first_BMW_indices_vec[0]), BMTB_first_BMW_indices_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMTB_first_BMW_indices_item(new meta_data_item(BMTB_first_BMW_indices_ptr, TBLOCK_META, "first_BMW_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMTB_first_BMW_indices_item);

    this->is_run = true;
}

// 输入数据只有BMTB的行索引
vector<shared_ptr<data_item_record>> get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(target_matrix_id >= 0);
    assert(this->is_run == true);

    vector<shared_ptr<data_item_record>> record_vec;

    // TBLOCK_META, "first_row_indices", this->target_matrix_id
    shared_ptr<data_item_record> BMTB_first_row_indices_record(new data_item_record(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    record_vec.push_back(BMTB_first_row_indices_record);

    shared_ptr<data_item_record> first_row_indices_of_BMW_record(new data_item_record(WARP_META, "first_row_indices", this->target_matrix_id));
    record_vec.push_back(first_row_indices_of_BMW_record);
    return record_vec;
}

vector<shared_ptr<data_item_record>> get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(target_matrix_id >= 0);
    assert(this->is_run == true);

    vector<shared_ptr<data_item_record>> record_vec;

    // TBLOCK_META, "first_BMW_indices", this->target_matrix_id
    shared_ptr<data_item_record> BMTB_first_BMW_indices_record(new data_item_record(TBLOCK_META, "first_BMW_indices", this->target_matrix_id));
    record_vec.push_back(BMTB_first_BMW_indices_record);

    return record_vec;
}

string get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "get_begin_BMWs_of_BMTB_after_blocking_in_row_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + "}";

    return return_str;
}
