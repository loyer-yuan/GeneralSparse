#include "../data_transform_step.hpp"

reorder_row_by_index::reorder_row_by_index(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("reorder_row_by_index", meta_data_set_ptr)
{
    this->target_matrix_id = target_matrix_id;
    assert(this->target_matrix_id >= 0);
}

void reorder_row_by_index::run(bool check)
{
    if (check)
    {
        // 保证metadata set
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        // 保证矩阵号的正确
        assert(this->target_matrix_id >= 0);

        // 保证对应子块的行索引、列索引、值存在
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));

        // 保证对应子块的每一个对应行号的原始行索引是存在的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id));
        // 查看子块的起始行号和结束行号
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
    }

    // 读取子块的起始行号和结束行号
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 一共的行数量
    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 排序之后每一行所对应的旧行的索引
    shared_ptr<universal_array> origin_row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    if (check)
    {
        assert(row_num_of_sub_matrix == origin_row_index_ptr->get_len());
    }

    // 当前子矩阵的行索引
    shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 获取每一行的非零元数量
    vector<unsigned long> nz_number_of_rows = get_nnz_of_each_row_in_spec_range(row_index_ptr, 0, row_num_of_sub_matrix - 1, 0, row_index_ptr->get_len() - 1);

    // shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
    // shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

    // assert(row_index_ptr->get_len() == col_index_ptr->get_len());
    // assert(col_index_ptr->get_len() == val_ptr->get_len());

    // 将当前子矩阵的列索引和值放到一个二维数组中
    // vector<vector<unsigned long>> col_index_of_each_row(row_num_of_sub_matrix);
    // vector<vector<double>> val_of_each_row(row_num_of_sub_matrix);

    // 用一个变量检查行索引的增序
    // unsigned long last_row_index = 0;

    // // 遍历所有非零元
    // for (unsigned long nz_index = 0; nz_index < row_index_ptr->get_len(); nz_index++)
    // {
    //     // 获取当前的行索引
    //     unsigned long cur_row_index = row_index_ptr->read_integer_from_arr(nz_index);
    //     assert(last_row_index <= cur_row_index);
    //     last_row_index = cur_row_index;

    //     unsigned long cur_col_index = col_index_ptr->read_integer_from_arr(nz_index);
    //     double cur_val = val_ptr->read_float_from_arr(nz_index);

    //     // 将数据放到数组对应的位置
    //     col_index_of_each_row[cur_row_index].push_back(cur_col_index);

    //     // 每一行的列坐标都是自增的
    //     if (col_index_of_each_row[cur_row_index].size() >= 2)
    //     {
    //         // 最后一位大于倒数第二个
    //         assert(col_index_of_each_row[cur_row_index][col_index_of_each_row[cur_row_index].size() - 1] >= col_index_of_each_row[cur_row_index][col_index_of_each_row[cur_row_index].size() - 2]);
    //     }

    //     val_of_each_row[cur_row_index].push_back(cur_val);
    // }

    // 重排之后的行索引，一维数组。
    vector<unsigned long> sorted_row_index_vec;
    // vector<unsigned long> sorted_col_index_vec;
    // vector<double> sorted_val_vec;

    // 遍历新的行索引所对应的老的行索引
    for (unsigned long new_row_id = 0; new_row_id < origin_row_index_ptr->get_len(); new_row_id++)
    {
        // 查看当前行在原始矩阵中的行
        unsigned long old_row_id = origin_row_index_ptr->read_integer_from_arr(new_row_id);
        if (check)
        {
            assert(old_row_id < nz_number_of_rows.size());
        }
        // 将原始行的索引拷贝到新的索引中
        for (unsigned long index_inner_row = 0; index_inner_row < nz_number_of_rows[old_row_id]; index_inner_row++)
        {
            sorted_row_index_vec.push_back(new_row_id);
            // sorted_col_index_vec.push_back(col_index_of_each_row[old_row_id][index_inner_row]);
            // sorted_val_vec.push_back(val_of_each_row[old_row_id][index_inner_row]);
        }
    }

    // 获取当前值的类型
    // data_type val_type = val_ptr->get_data_type();

    // assert(val_type == FLOAT || val_type == DOUBLE);

    // 在metadata set中删除行、列、值三个索引
    meta_data_set_ptr->remove_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    // meta_data_set_ptr->remove_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    // meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

    // 重新创建三个索引
    shared_ptr<universal_array> new_row_indices_ptr(new universal_array(((void *)(&sorted_row_index_vec[0])), sorted_row_index_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> new_row_indices_item(new meta_data_item(new_row_indices_ptr, GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    meta_data_set_ptr->add_element(new_row_indices_item);

    this->is_run = true;

    // shared_ptr<universal_array> new_col_indices_ptr(new universal_array(((void *)(&sorted_col_index_vec[0])), sorted_col_index_vec.size(), UNSIGNED_LONG));
    // shared_ptr<meta_data_item> new_col_indices_item(new meta_data_item(new_col_indices_ptr, GLOBAL_META, "nz_col_indices", this->target_matrix_id));
    // meta_data_set_ptr->add_element(new_col_indices_item);

    // shared_ptr<universal_array> new_vals_ptr(new universal_array(((void *)(&sorted_val_vec[0])), sorted_val_vec.size(), val_type));
    // shared_ptr<meta_data_item> new_vals_item(new meta_data_item(new_vals_ptr, GLOBAL_META, "nz_vals", this->target_matrix_id));
    // meta_data_set_ptr->add_element(new_vals_item);
}

vector<shared_ptr<data_item_record>> reorder_row_by_index::get_source_data_item_ptr_in_data_transform_step()
{
    // input: nz_row_indices, original_nz_row_indices, GLOBAL_META.
    // assert(this->is_run == true); 排序的执行记录是可以静态得出的
    assert(this->target_matrix_id >= 0);
    vector<shared_ptr<data_item_record>> return_vec;

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    return_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> original_nz_row_indices_record(new data_item_record(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id));
    return_vec.push_back(original_nz_row_indices_record);

    return return_vec;
}

vector<shared_ptr<data_item_record>> reorder_row_by_index::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // output: nz_row_indices, GLOBAL_META.
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true); 排序的执行记录是可以静态得出的

    vector<shared_ptr<data_item_record>> return_vec;

    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));

    return_vec.push_back(nz_row_indices_record);

    return return_vec;
}

string reorder_row_by_index::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    // 打印名字和参数
    string return_str = "reorder_row_by_index::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}