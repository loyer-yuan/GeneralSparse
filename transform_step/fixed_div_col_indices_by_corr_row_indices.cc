#include "data_transform_step.hpp"

// 构造函数
fixed_div_col_indices_by_corr_row_indices::fixed_div_col_indices_by_corr_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size)
    : basic_data_transform_step("fixed_div_col_indices_by_corr_row_indices", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_gap_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_gap_size = fixed_row_gap_size;
}

// 执行当前的data transform step
void fixed_div_col_indices_by_corr_row_indices::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->fixed_row_gap_size > 0);

        // 首先分割列索引，并且这个分割基于特定子块列索引和行索引的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // 列索引也必须存在
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
        // 查看当前子块的行范围，这是就不是做检查了，而是确定桶的数量，是重要输入
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

    // 行索引
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> nz_col_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录两个输入记录
    shared_ptr<data_item_record> nz_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    shared_ptr<data_item_record> nz_col_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record_ptr);
    this->source_data_item_ptr_vec.push_back(nz_col_indices_record_ptr);

    if (check)
    {
        assert(end_row_index >= begin_row_index);
        assert(nz_row_indices_ptr->get_len() == nz_col_indices_ptr->get_len());
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

    // 使用一个二维数组来存储分块之后的列索引
    vector<vector<unsigned long>> nz_col_indices_in_each_sub_matrix(sub_matrix_num);

    // 遍历所有非零元的列索引，以及其对应的行索引
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        // 一个非零元的列索引和行索引
        unsigned long cur_col_index = nz_col_indices_ptr->read_integer_from_arr(i);
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);

        // 通过行索引算出来当前行索引所属于的子矩阵
        unsigned long cur_sub_matrix_id = cur_row_index / this->get_fixed_row_gap_size();
        if (check)
        {
            assert(cur_sub_matrix_id < sub_matrix_num);
        }
        nz_col_indices_in_each_sub_matrix[cur_sub_matrix_id].push_back(cur_col_index);
    }

    // 建立遍历所有的子块，为根据每一个子块建立一个新的列索引数据
    for (unsigned long i = 0; i < nz_col_indices_in_each_sub_matrix.size(); i++)
    {
        // 如果当前的行条带是存在的，就要创造新的列索引
        if (nz_col_indices_in_each_sub_matrix[i].size() != 0)
        {
            // 增加新的子块，查看已有列索引的最大子矩阵索引，查看当前子块的最大子矩阵号
            int max_existing_sub_matrix_id_of_this_item = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_col_indices");

            // cout << "max_existing_sub_matrix_id_of_this_item:" << max_existing_sub_matrix_id_of_this_item << endl;

            // 当前的子矩阵的列索引
            vector<unsigned long> col_indice_of_this_sub_matrix = nz_col_indices_in_each_sub_matrix[i];

            // 创造一个新的列索引
            shared_ptr<universal_array> new_col_indices_ptr(new universal_array(((void *)(&col_indice_of_this_sub_matrix[0])), col_indice_of_this_sub_matrix.size(), UNSIGNED_LONG));
            // 创造metadata set表项，新的子块编号是最大子矩阵号+1
            shared_ptr<meta_data_item> new_col_indices_item_ptr(new meta_data_item(new_col_indices_ptr, GLOBAL_META, "nz_col_indices", max_existing_sub_matrix_id_of_this_item + 1));
            this->meta_data_set_ptr->add_element(new_col_indices_item_ptr);
            // 记录当前的输出
            shared_ptr<data_item_record> new_col_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_col_indices", max_existing_sub_matrix_id_of_this_item + 1));
            this->dest_data_item_ptr_vec.push_back(new_col_indices_record_ptr);
        }
    }

    // 删除列索引
    this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id);

    // 记录已经执行过了
    this->is_run = true;
}

// 给出当前data transform step的输入记录
vector<shared_ptr<data_item_record>> fixed_div_col_indices_by_corr_row_indices::get_source_data_item_ptr_in_data_transform_step()
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
vector<shared_ptr<data_item_record>> fixed_div_col_indices_by_corr_row_indices::get_dest_data_item_ptr_in_data_transform_step_without_check()
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
string fixed_div_col_indices_by_corr_row_indices::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    // 打印参数
    string return_str = "div_col_indices_by_corr_row_indices::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",fixed_row_gap_size:" + to_string(this->fixed_row_gap_size) + "}";

    return return_str;
}