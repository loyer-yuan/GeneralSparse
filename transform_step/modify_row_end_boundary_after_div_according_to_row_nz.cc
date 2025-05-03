#include "data_transform_step.hpp"

modify_row_end_boundary_after_div_according_to_row_nz::modify_row_end_boundary_after_div_according_to_row_nz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate)
    : basic_data_transform_step("modify_row_end_boundary_after_div_according_to_row_nz", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(nz_gap_size > 0);

    this->target_matrix_id = target_matrix_id;
    this->nz_gap_size = nz_gap_size;
    this->max_gap = max_gap;
    this->expansion_rate = expansion_rate;
}

void modify_row_end_boundary_after_div_according_to_row_nz::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->nz_gap_size > 0);
        assert(this->max_gap >= this->nz_gap_size);
        assert(this->expansion_rate > 1);

        // 目标子矩阵的行边界都是存在的
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 把行索读出来，发现潜在的空条带
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    // 读出来当前子块的首行索引和最后一行的索引
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 输入数据的记录
    shared_ptr<data_item_record> begin_row_index_record_ptr(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
    shared_ptr<data_item_record> end_row_index_record_ptr(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(begin_row_index_record_ptr);
    this->source_data_item_ptr_vec.push_back(end_row_index_record_ptr);

    // 读出行索引，查看空白行
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 记录行索引
    shared_ptr<data_item_record> nz_row_indices_record_ptr(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record_ptr);

    if (check)
    {
        assert(end_row_index >= begin_row_index);
    }

    // 查看当前行的数量
    unsigned long row_num = end_row_index - begin_row_index + 1;

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
    div_position.push_back(nnz_of_each_row.size());

    // 遍历所有桶，如果不是空桶，那就使用
    for (unsigned long i = 1; i < div_position.size(); i++)
    {

        // 获取子矩阵行边界的最大子id
        int max_existing_sub_matrix_id_of_end_row_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "end_row_index");

        // 新的子矩阵的首行索引
        unsigned long new_end_row_index = div_position[i] - 1;

        if (check)
        {
            // 加入之前肯定不存在
            assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", max_existing_sub_matrix_id_of_end_row_index + 1) == false);
        }

        shared_ptr<meta_data_item> end_row_item(new meta_data_item(((void *)(&new_end_row_index)), UNSIGNED_LONG, GLOBAL_META, "end_row_index", max_existing_sub_matrix_id_of_end_row_index + 1));

        // 加入
        this->meta_data_set_ptr->add_element(end_row_item);

        // 增加一个输出记录
        shared_ptr<data_item_record> new_end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", max_existing_sub_matrix_id_of_end_row_index + 1));
        this->dest_data_item_ptr_vec.push_back(new_end_row_index_record);
    }

    // 记录已经执行过了
    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_row_end_boundary_after_div_according_to_row_nz::get_source_data_item_ptr_in_data_transform_step()
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

vector<shared_ptr<data_item_record>> modify_row_end_boundary_after_div_according_to_row_nz::get_dest_data_item_ptr_in_data_transform_step_without_check()
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
string modify_row_end_boundary_after_div_according_to_row_nz::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    // 打印参数
    string return_str = "modify_row_end_boundary_after_div_according_to_row_nz::{name:\"" + this->name + "\",target_matrix_id:" + to_string(this->target_matrix_id) + ",nz_gap_size:" + to_string(this->nz_gap_size) + "}";

    return return_str;
}