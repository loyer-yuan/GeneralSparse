#include "data_transform_step.hpp"

// 构造函数
fixed_div_vals_by_corr_row_indices::fixed_div_vals_by_corr_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size)
    : basic_data_transform_step("fixed_div_vals_by_corr_row_indices", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(fixed_row_gap_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->fixed_row_gap_size = fixed_row_gap_size;
}

// 执行当前的data transform step
void fixed_div_vals_by_corr_row_indices::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->fixed_row_gap_size > 0);

        // 首先分割列索引，并且这个分割基于特定子块列索引和行索引的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // 值也必须存在
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id));
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
    shared_ptr<universal_array> nz_vals_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

    // 记录两个输入
    shared_ptr<data_item_record> nz_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    shared_ptr<data_item_record> nz_vals_record_ptr(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record_ptr);
    this->source_data_item_ptr_vec.push_back(nz_vals_record_ptr);
    data_type val_data_type = nz_vals_ptr->get_data_type();

    if (check)
    {
        assert(nz_row_indices_ptr->get_len() == nz_vals_ptr->get_len());
        assert(end_row_index >= begin_row_index);
        // 值的数据类型
        assert(val_data_type == FLOAT || val_data_type == DOUBLE);
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

    // 使用一个二维数组来存储分块之后的值数组
    vector<vector<double>> nz_vals_indices_in_each_sub_matrix(sub_matrix_num);

    // 遍历所有非零元的行索引和值，并且对值进行分块
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        // 一个非零元的行索引和值
        double cur_val = nz_vals_ptr->read_float_from_arr(i);
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);

        // 当前所属的矩阵
        unsigned long cur_sub_matrix_id = cur_row_index / this->get_fixed_row_gap_size();
        if (check)
        {
            assert(cur_sub_matrix_id < sub_matrix_num);
        }
        nz_vals_indices_in_each_sub_matrix[cur_sub_matrix_id].push_back(cur_val);
    }

    // 遍历所有子块，分别为每一个子块建立一个值数组
    for (unsigned long i = 0; i < nz_vals_indices_in_each_sub_matrix.size(); i++)
    {
        // 如果当前的行条带是存在的，就要创造新的值数组
        if (nz_vals_indices_in_each_sub_matrix[i].size() != 0)
        {
            // 查看当前值数组所在的最大子矩阵号
            int max_existing_sub_matrix_id_of_this_item = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_vals");

            // 当前子矩阵的值
            vector<double> vals_of_this_sub_matrix = nz_vals_indices_in_each_sub_matrix[i];

            // 创造一个新的值数组
            shared_ptr<universal_array> new_vals_ptr(new universal_array(((void *)(&vals_of_this_sub_matrix[0])), vals_of_this_sub_matrix.size(), DOUBLE));

            // 如果是单精度那就要压缩
            if (val_data_type == FLOAT)
            {
                new_vals_ptr->compress_float_precise();
                if (check)
                {
                    assert(new_vals_ptr->get_data_type() == FLOAT);
                }
            }

            // 创造metadata set表项，新的子块编号是最大子矩阵号+1
            shared_ptr<meta_data_item> new_vals_item_ptr(new meta_data_item(new_vals_ptr, GLOBAL_META, "nz_vals", max_existing_sub_matrix_id_of_this_item + 1));
            this->meta_data_set_ptr->add_element(new_vals_item_ptr);
            // 记录当前输出
            shared_ptr<data_item_record> new_vals_record_ptr(new data_item_record(GLOBAL_META, "nz_vals", max_existing_sub_matrix_id_of_this_item + 1));
            this->dest_data_item_ptr_vec.push_back(new_vals_record_ptr);
        }
    }

    // 删除值数组
    this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

    // 记录已经执行过了
    this->is_run = true;
}

vector<shared_ptr<data_item_record>> fixed_div_vals_by_corr_row_indices::get_source_data_item_ptr_in_data_transform_step()
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
vector<shared_ptr<data_item_record>> fixed_div_vals_by_corr_row_indices::get_dest_data_item_ptr_in_data_transform_step_without_check()
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
string fixed_div_vals_by_corr_row_indices::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    // 打印参数
    string return_str = "fixed_div_vals_by_corr_row_indices::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",fixed_row_gap_size:" + to_string(this->fixed_row_gap_size) + "}";

    return return_str;
}