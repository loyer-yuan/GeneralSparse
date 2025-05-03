#include "data_transform_step.hpp"

// 构造函数
div_row_indices_by_row_nnz::div_row_indices_by_row_nnz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate)
    : basic_data_transform_step("div_row_indices_by_row_nnz", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(nz_gap_size > 0);
    assert(expansion_rate > 1);

    this->target_matrix_id = target_matrix_id;
    this->nz_gap_size = nz_gap_size;
    this->max_gap = max_gap;
    this->expansion_rate = expansion_rate;
}

// 执行当前的data transform step
void div_row_indices_by_row_nnz::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->nz_gap_size > 0);
        assert(this->max_gap >= this->nz_gap_size);
        assert(this->expansion_rate > 1);

        // 首先分割列索引，并且这个分割基于特定子块列索引和行索引的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // 列索引也必须存在
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

    // 记录两个输入记录
    shared_ptr<data_item_record> nz_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record_ptr);

    if (check)
    {
        assert(end_row_index >= begin_row_index);
    }

    unsigned long row_num = end_row_index - begin_row_index + 1;
    // 查看当前行的数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    // 遍历所有行数据
    vector<unsigned long> div_position;
    // 用一个变量来存储当前行的行非零元上界和下界，只要在这个界限之外，就代表需要需要开一个新的分块点了
    // 上界是不被包含的，下界是被包含的
    unsigned long row_index_low_bound_of_cur_window = 0;
    unsigned long row_index_high_bound_of_cur_window = this->nz_gap_size;

    unsigned long cur_row_nz = nnz_of_each_row[0];
    div_position.push_back(0);

    // 找到对应子窗口，这里代表没有找到对应的窗口
    if (cur_row_nz < this->max_gap)
    {
        while (cur_row_nz >= row_index_high_bound_of_cur_window && row_index_high_bound_of_cur_window <= this->max_gap)
        {
            // 在这里代表没有找到对应的行非零元范围的窗口，需要重新调整窗口位置
            row_index_low_bound_of_cur_window = row_index_high_bound_of_cur_window;
            row_index_high_bound_of_cur_window = row_index_high_bound_of_cur_window * expansion_rate;
        }
    }
    else
    {
        row_index_low_bound_of_cur_window = this->max_gap;
        row_index_high_bound_of_cur_window = row_index_low_bound_of_cur_window * expansion_rate;
    }

    // 在这里row_index_low_bound_of_cur_window和row_index_high_bound_of_cur_window得到了第一个窗口。
    // 遍历剩下的行，每当不在之前的区间之内就记录一个新的分块点
    for (unsigned long row_id = 1; row_id < nnz_of_each_row.size(); row_id++)
    {
        // 新的行非零元数量
        cur_row_nz = nnz_of_each_row[row_id];

        if (cur_row_nz >= row_index_low_bound_of_cur_window && cur_row_nz < row_index_high_bound_of_cur_window || (cur_row_nz >= row_index_high_bound_of_cur_window && row_index_high_bound_of_cur_window > this->max_gap))
        {
            // 这里代表当前行的窗口和上一行是一致的
            continue;
        }

        // 这里代表到达了新的行非零元窗口范围，记录为一个块的首行索引，然后然后找出当前行所在的区间
        div_position.push_back(row_id);
        row_index_low_bound_of_cur_window = 0;
        row_index_high_bound_of_cur_window = this->nz_gap_size;

        // 找到对应的新的非零元数量的窗口
        if (cur_row_nz < this->max_gap)
        {
            while (cur_row_nz >= row_index_high_bound_of_cur_window && row_index_high_bound_of_cur_window <= this->max_gap)
            {
                // 在这里代表没有找到对应的行非零元范围的窗口，需要重新调整窗口位置
                row_index_low_bound_of_cur_window = row_index_high_bound_of_cur_window;
                row_index_high_bound_of_cur_window = row_index_high_bound_of_cur_window * expansion_rate;
            }
        }
        else
        {
            row_index_low_bound_of_cur_window = this->max_gap;
            row_index_high_bound_of_cur_window = row_index_low_bound_of_cur_window * expansion_rate;
        }
    }

    // 使用一个二维数组来存储分块之后的列索引
    vector<vector<unsigned long>> nz_row_indices_in_each_sub_matrix(div_position.size());
    unsigned long cur_sub_matrix_id = 0;

    // 遍历所有非零元的列索引，以及其对应的行索引
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        // 一个非零元的列索引和行索引
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);
        int upper_bound = (cur_sub_matrix_id < div_position.size() - 1) ? div_position[cur_sub_matrix_id + 1] : nz_row_indices_ptr->read_integer_from_arr(nz_row_indices_ptr->get_len() - 1) + 1;
        if (cur_row_index >= div_position[cur_sub_matrix_id] && cur_row_index < upper_bound)
        {
            nz_row_indices_in_each_sub_matrix[cur_sub_matrix_id].push_back(cur_row_index);
        }
        else if (cur_row_index >= upper_bound)
        {
            cur_sub_matrix_id += 1;
            nz_row_indices_in_each_sub_matrix[cur_sub_matrix_id].push_back(cur_row_index);
        }
    }

    // 建立遍历所有的子块，为根据每一个子块建立一个新的列索引数据
    for (unsigned long i = 0; i < nz_row_indices_in_each_sub_matrix.size(); i++)
    {
        // 如果当前的行条带是存在的，就要创造新的列索引
        if (nz_row_indices_in_each_sub_matrix[i].size() != 0)
        {
            // 增加新的子块，查看已有列索引的最大子矩阵索引，查看当前子块的最大子矩阵号
            int max_existing_sub_matrix_id_of_this_item = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_row_indices");

            // 当前的子矩阵的列索引
            vector<unsigned long> row_indice_of_this_sub_matrix = nz_row_indices_in_each_sub_matrix[i];

            // 创造一个新的列索引
            shared_ptr<universal_array> new_row_indices_ptr(new universal_array(((void *)(&row_indice_of_this_sub_matrix[0])), row_indice_of_this_sub_matrix.size(), UNSIGNED_LONG));
            // 创造metadata set表项，新的子块编号是最大子矩阵号+1
            shared_ptr<meta_data_item> new_row_indices_item_ptr(new meta_data_item(new_row_indices_ptr, GLOBAL_META, "nz_row_indices", max_existing_sub_matrix_id_of_this_item + 1));
            this->meta_data_set_ptr->add_element(new_row_indices_item_ptr);
            // 记录当前的输出
            shared_ptr<data_item_record> new_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", max_existing_sub_matrix_id_of_this_item + 1));
            this->dest_data_item_ptr_vec.push_back(new_row_indices_record_ptr);
        }
    }

    // 删除列索引
    this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id);

    // 记录已经执行过了
    this->is_run = true;
}

// 给出当前data transform step的输入记录
vector<shared_ptr<data_item_record>> div_row_indices_by_row_nnz::get_source_data_item_ptr_in_data_transform_step()
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
vector<shared_ptr<data_item_record>> div_row_indices_by_row_nnz::get_dest_data_item_ptr_in_data_transform_step_without_check()
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
string div_row_indices_by_row_nnz::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    // 打印参数
    string return_str = "div_row_indices_by_row_nnz::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",nz_gap_size:" + to_string(this->nz_gap_size) + "}";

    return return_str;
}