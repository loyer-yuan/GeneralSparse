#include "data_transform_step.hpp"
// test
fixed_div_row_indices::fixed_div_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size)
    : basic_data_transform_step("fixed_div_row_indices", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_gap_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_gap_size = fixed_row_gap_size;
}

// 执行当前的data transform step
void fixed_div_row_indices::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->fixed_row_gap_size > 0);

        // 需要行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // 需要当前子块的行索引范围
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
    }

    // 读出来当前子块的首行索引和最后一行的索引
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 输入数据的记录
    shared_ptr<data_item_record> begin_row_index_record_ptr(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    shared_ptr<data_item_record> end_row_index_record_ptr(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(begin_row_index_record_ptr);
    this->source_data_item_ptr_vec.push_back(end_row_index_record_ptr);
    if (check)
    {
        assert(end_row_index >= begin_row_index);
    }

    // 查看当前行的数量
    unsigned long row_num = end_row_index - begin_row_index + 1;

    // 查看当前的行数量要分多少个子块
    unsigned long sub_matrix_num = row_num / this->get_fixed_row_gap_size();

    // 如果不能整除，那么就需要多一个子块
    if (row_num % this->get_fixed_row_gap_size() != 0)
    {
        sub_matrix_num = sub_matrix_num + 1;
    }

    // 行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录当行索引输入
    shared_ptr<data_item_record> nz_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record_ptr);

    // 使用一个二维数组来存储
    vector<vector<unsigned long>> nz_row_indices_in_each_sub_matrix(sub_matrix_num);

    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        // 一个非零元的行索引
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);

        // 当前所属的矩阵
        unsigned long cur_sub_matrix_id = cur_row_index / this->get_fixed_row_gap_size();

        // 将索引换成相对索引
        cur_row_index = cur_row_index % this->get_fixed_row_gap_size();
        if (check)
        {
            assert(cur_sub_matrix_id < sub_matrix_num);
        }
        nz_row_indices_in_each_sub_matrix[cur_sub_matrix_id].push_back(cur_row_index);
    }

    // 遍历所有子块，分别为每一个子块建立一个行索引，并存在metadata set中
    for (unsigned long i = 0; i < nz_row_indices_in_each_sub_matrix.size(); i++)
    {
        // 如果当前的行条带是存在的，就要创造新的值数组
        if (nz_row_indices_in_each_sub_matrix[i].size() != 0)
        {
            // 查看当前行索引所在的最大子矩阵号
            int max_existing_sub_matrix_id_of_this_item = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_row_indices");

            // 当前子矩阵的行索引
            vector<unsigned long> row_indices_of_this_sub_matrix = nz_row_indices_in_each_sub_matrix[i];

            // 创造一个行索引
            shared_ptr<universal_array> new_row_indices_ptr(new universal_array(((void *)(&row_indices_of_this_sub_matrix[0])), row_indices_of_this_sub_matrix.size(), UNSIGNED_LONG));

            // 创造metadata set中的表项
            shared_ptr<meta_data_item> new_row_indices_item_ptr(new meta_data_item(new_row_indices_ptr, GLOBAL_META, "nz_row_indices", max_existing_sub_matrix_id_of_this_item + 1));
            this->meta_data_set_ptr->add_element(new_row_indices_item_ptr);
            // 记录当前输出
            shared_ptr<data_item_record> new_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", max_existing_sub_matrix_id_of_this_item + 1));
            this->dest_data_item_ptr_vec.push_back(new_row_indices_record_ptr);
        }
    }

    // 删除行索引
    this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);

    // 记录已经执行过了
    this->is_run = true;
}

vector<shared_ptr<data_item_record>> fixed_div_row_indices::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    // 检查输出数据
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(source_data_item_ptr_vec[i] != NULL);
    }

    // 返回输出数据
    return this->source_data_item_ptr_vec;
}

// 给出当前data transform step的输出记录
vector<shared_ptr<data_item_record>> fixed_div_row_indices::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

// 转化为字符串
string fixed_div_row_indices::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    // 打印参数
    string return_str = "fixed_div_row_indices::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",fixed_row_gap_size:" + to_string(this->fixed_row_gap_size) + "}";

    return return_str;
}