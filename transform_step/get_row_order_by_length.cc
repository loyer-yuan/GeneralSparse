#include "../data_transform_step.hpp"

get_row_order_by_length::get_row_order_by_length(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("get_row_order_by_length", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    // 没有数据成员需要处理
    // 仅仅做一些检查
    this->target_matrix_id = target_matrix_id;
}

void get_row_order_by_length::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        // 检查
        assert(this->meta_data_set_ptr->check());
        // 查看对应的数据是不是存在的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // 已经排过序了就不能再排序了
        assert(!(this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id)));
    }

    // 将条目读出来，获得的是当前子矩阵的索引
    shared_ptr<meta_data_item> item = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);

    // 将行索引条目读出来
    shared_ptr<universal_array> row_index = item->get_metadata_arr();

    // 获取当前子矩阵理论上的最大行索引，考虑有空行
    unsigned long min_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long max_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 针对可能的行方向padding，找到真正的最大行索引
    unsigned long real_max_row_index = row_index->read_integer_from_arr(row_index->get_len() - 1);

    if (check)
    {
        assert(max_row_index >= min_row_index);
    }

    unsigned long max_logic_relative_row_index = max_row_index - min_row_index;

    if (real_max_row_index > max_logic_relative_row_index)
    {
        max_row_index = min_row_index + real_max_row_index;
    }

    unsigned long relative_max_row_index = max_row_index - min_row_index;

    // 获得每一行的非零元数量，
    vector<unsigned long> row_nz_number = get_nnz_of_each_row_in_spec_range(row_index, 0, relative_max_row_index, 0, row_index->get_len() - 1);

    // 行数量
    if (check)
    {
        assert(row_nz_number.size() == relative_max_row_index + 1);
    }

    // 获取行非零元数量的最大值
    unsigned long max_row_length = *max_element(row_nz_number.begin(), row_nz_number.end());

    // 使用一个数组使用桶排序，桶的数量为最大的行长度，每个桶是一个数组，将特定行长度的行索引放到对应的桶中。
    vector<vector<unsigned long>> bin_of_diff_row_size(max_row_length + 1);

    // 遍历所有的行非零元数量
    for (unsigned long cur_row = 0; cur_row < row_nz_number.size(); cur_row++)
    {
        unsigned long cur_row_len = row_nz_number[cur_row];

        // 将行号放到对应的桶的末尾
        bin_of_diff_row_size[cur_row_len].push_back(cur_row);
    }

    // 倒着遍历不同行长度的行索引对应的桶，将经过降序排列行索引弄出来，这个算子仅仅产出排序之后的行和原始行索引的映射。对于真正的排序，可以参考total_dense_block_coarse_sort和total_dense_block_sort两个函数
    // 将bin_of_diff_row_size拉平，放到一个新的一维数组中
    vector<unsigned long> row_index_order_by_length_vec;

    // 倒序，哨兵变量要能支持负数
    for (long bin_id = bin_of_diff_row_size.size() - 1; bin_id >= 0; bin_id--)
    {
        // cout << bin_id << endl;
        // 遍历每个桶的内部
        for (long inner_row_index_id = 0; inner_row_index_id < bin_of_diff_row_size[bin_id].size(); inner_row_index_id++)
        {
            row_index_order_by_length_vec.push_back(bin_of_diff_row_size[bin_id][inner_row_index_id]);

            // if (row_index_order_by_length_vec.size() < 10)
            // {
            //     cout << "get_row_order_by_length::run():" << bin_of_diff_row_size[bin_id][inner_row_index_id] << endl;
            // }

            // 检查看看行非零元数量是不是降序
            if (row_index_order_by_length_vec.size() >= 2)
            {
                // 当前插入的行号
                unsigned long cur_row_index = row_index_order_by_length_vec[row_index_order_by_length_vec.size() - 1];
                unsigned long prev_row_index = row_index_order_by_length_vec[row_index_order_by_length_vec.size() - 2];
                // 保证降序
                if (check)
                {
                    assert(row_nz_number[cur_row_index] <= row_nz_number[prev_row_index]);
                }
            }
        }
    }

    // 将一维数组中的内容转化为metadata set中的origin row index
    if (check)
    {
        assert(row_nz_number.size() == row_index_order_by_length_vec.size());
    }

    // 首先创造一个通用数组
    shared_ptr<universal_array> original_row_index_ptr(new universal_array((void *)(&row_index_order_by_length_vec[0]), row_index_order_by_length_vec.size(), UNSIGNED_LONG));

    // 将数组中的内容放到metadata set中，先产生一个条目
    shared_ptr<meta_data_item> item_ptr(new meta_data_item(original_row_index_ptr, GLOBAL_META, "original_nz_row_indices", this->target_matrix_id));

    // 加入到元数据集中
    this->meta_data_set_ptr->add_element(item_ptr);
    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_row_order_by_length::get_source_data_item_ptr_in_data_transform_step()
{
    // input1:GLOBAL_META, "nz_row_indices", this->target_matrix_id
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true); 这里的执行的输入和输出可以在静态的方式中进行
    vector<shared_ptr<data_item_record>> return_vec;
    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));

    return_vec.push_back(nz_row_indices_record);

    return return_vec;
}

vector<shared_ptr<data_item_record>> get_row_order_by_length::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // output2:GLOBAL_META, "original_nz_row_indices", this->target_matrix_id
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true); 这里的执行的输入和输出可以在静态的方式中进行

    vector<shared_ptr<data_item_record>> return_vec;

    shared_ptr<data_item_record> original_nz_row_indices_record(new data_item_record(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id));

    return_vec.push_back(original_nz_row_indices_record);

    return return_vec;
}

string get_row_order_by_length::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    // 打印名字和参数
    string return_str = "get_row_order_by_length::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}