#include "../operator.hpp"

// 初始化
row_nz_matrix_div_operator::row_nz_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int init_row_size_upper_boundary, int max_row_size_upper_boundary, int expansion_rate)
    : basic_operator("row_nz_matrix_div_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(init_row_size_upper_boundary > 0);
    assert(expansion_rate > 0);

    this->init_row_size_upper_boundary = init_row_size_upper_boundary;
    this->max_row_size_upper_boundary = max_row_size_upper_boundary;
    this->expansion_rate = expansion_rate;
}

row_nz_matrix_div_operator::row_nz_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int init_row_size_upper_boundary, int max_row_size_upper_boundary, int expansion_rate, shared_ptr<operator_context> operator_history)
    : basic_operator("row_nz_matrix_div_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(init_row_size_upper_boundary > 0);
    assert(expansion_rate > 0);

    this->init_row_size_upper_boundary = init_row_size_upper_boundary;
    this->max_row_size_upper_boundary = max_row_size_upper_boundary;
    this->expansion_rate = expansion_rate;
}

row_nz_matrix_div_operator::row_nz_matrix_div_operator(shared_ptr<code_generator> code_generator_ptr, int init_row_size_upper_boundary, int max_row_size_upper_boundary, int expansion_rate, shared_ptr<operator_context> operator_history)
    : basic_operator("row_nz_matrix_div_operator", code_generator_ptr->get_metadata_set(), CONVERTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this)row_nz_matrix_div_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), init_row_size_upper_boundary, max_row_size_upper_boundary, expansion_rate, operator_history);
}

bool row_nz_matrix_div_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_converting = operator_history->read_operator_context_arr(CONVERTING_OP, this->target_matrix_id);
    bool self_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        if (former_operator_distributing.size() == 0)
        {
            for (int i = 0; i < former_operator_converting.size(); i++)
            {
                if (former_operator_converting[i]->get_name().find("div_operator") != string::npos)
                {
                    if (former_operator_converting[i]->get_target_matrix_id() == this->get_target_matrix_id())
                    {
                        self_flag = true;
                    }
                }
            }
        }
    }
    if (self_flag == false)
    {
        return true;
    }

    return false;
}

bool row_nz_matrix_div_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);

    // 目标子矩阵的所有内容（行列边界，行列值数组）
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);
    bool start_col_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_col_index", this->target_matrix_id);
    bool end_col_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_col_index", this->target_matrix_id);
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);

    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    bool div_count_flag = true;

    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 查看当前行的数量
    unsigned long row_num = end_row_index - begin_row_index + 1;

    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(nz_row_indices_ptr, 0, row_num - 1, 0, nz_row_indices_ptr->get_len() - 1);

    // 遍历所有行数据
    vector<unsigned long> div_position;

    unsigned long row_index_low_bound_of_cur_window = 0;
    unsigned long row_index_high_bound_of_cur_window = this->init_row_size_upper_boundary;

    unsigned long cur_row_nz = nnz_of_each_row[0];
    div_position.push_back(0);

    // 先预先尝试进行切分，查看切分出的子矩阵是不是超过了要求
    // 先保证行非零元数量在
    if (cur_row_nz < this->max_row_size_upper_boundary)
    {
        while (cur_row_nz >= row_index_high_bound_of_cur_window && row_index_high_bound_of_cur_window <= this->max_row_size_upper_boundary)
        {
            // 在这里代表没有找到对应的行非零元范围的窗口，需要重新调整窗口位置
            row_index_low_bound_of_cur_window = row_index_high_bound_of_cur_window;
            row_index_high_bound_of_cur_window = row_index_high_bound_of_cur_window * expansion_rate;
        }
    }
    else
    {
        row_index_low_bound_of_cur_window = this->max_row_size_upper_boundary;
        row_index_high_bound_of_cur_window = row_index_low_bound_of_cur_window* expansion_rate;
    }

    // 在这里row_index_low_bound_of_cur_window和row_index_high_bound_of_cur_window得到了第一个窗口。
    // 遍历剩下的行，每当不在之前的区间之内就记录一个新的分块点
    for (unsigned long row_id = 1; row_id < nnz_of_each_row.size(); row_id++)
    {
        // 新的行非零元数量
        cur_row_nz = nnz_of_each_row[row_id];

        if (cur_row_nz >= row_index_low_bound_of_cur_window && cur_row_nz < row_index_high_bound_of_cur_window || (cur_row_nz >= row_index_high_bound_of_cur_window && row_index_high_bound_of_cur_window > this->max_row_size_upper_boundary))
        {
            // 这里代表当前行的窗口和上一行是一致的
            continue;
        }

        // 这里代表到达了新的行非零元窗口范围，记录为一个块的首行索引，然后然后找出当前行所在的区间
        div_position.push_back(row_id);

        // 切分数量超出限制范围，则不进行矩阵分块
        if (div_position.size() > get_config()["MAX_DIV_TIMES_OF_DIV"].as_integer())
        {
            div_count_flag = false;
            break;
        }
        // 初始化窗口
        row_index_low_bound_of_cur_window = 0;
        row_index_high_bound_of_cur_window = this->init_row_size_upper_boundary;

        // 找到对应的新的非零元数量的窗口
        if (cur_row_nz < this->max_row_size_upper_boundary)
        {
            while (cur_row_nz >= row_index_high_bound_of_cur_window && row_index_high_bound_of_cur_window <= this->max_row_size_upper_boundary)
            {
                // 在这里代表没有找到对应的行非零元范围的窗口，需要重新调整窗口位置
                row_index_low_bound_of_cur_window = row_index_high_bound_of_cur_window;
                row_index_high_bound_of_cur_window = row_index_high_bound_of_cur_window * expansion_rate;
            }
        }
        else
        {
            row_index_low_bound_of_cur_window = this->max_row_size_upper_boundary;
            row_index_high_bound_of_cur_window = row_index_low_bound_of_cur_window * expansion_rate;
        }
    }


    // 上述内容全部存在才能执行
    if (start_row_boundary == true && end_row_boundary == true && start_col_boundary == true && end_col_boundary == true && interlance_storage_existing == false)
    {
        if (row_indices_existing == true && col_indices_existing == true && vals_existing == true && div_count_flag == true)
        {
            return true;
        }
    }

    return false;
}

void row_nz_matrix_div_operator::run(bool check)
{
    if (check)
    {
        assert(this->is_valid_according_to_metadata());
    }
    // 查看当前能不能执行，之前要通过这个检查，不能通过那就直接退出

    int max_existing_sub_matrix_id_of_end_row_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "end_row_index");
    int max_existing_sub_matrix_id_of_begin_row_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "begin_row_index");
    int max_existing_sub_matrix_id_of_begin_col_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "begin_col_index");
    int max_existing_sub_matrix_id_of_end_col_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "end_col_index");
    int max_existing_sub_matrix_id_of_nz_row_indices = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_row_indices");
    int max_existing_sub_matrix_id_of_nz_col_indices = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_col_indices");
    int max_existing_sub_matrix_id_of_nz_vals = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_vals");
    if (check)
    { // 各种数据的最大子矩阵号，做一个检查
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_begin_row_index);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_begin_col_index);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_end_col_index);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_nz_row_indices);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_nz_col_indices);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_nz_vals);
    }
    int last_max_sub_matrix_id = max_existing_sub_matrix_id_of_end_row_index;

    // 首先修改行列边界
    shared_ptr<modify_row_start_boundary_after_div_according_to_row_nz> modify_row_start_boundary_after_div_according_to_row_nz_ptr(new modify_row_start_boundary_after_div_according_to_row_nz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    modify_row_start_boundary_after_div_according_to_row_nz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_start_boundary_after_div_according_to_row_nz_ptr));

    shared_ptr<modify_row_end_boundary_after_div_according_to_row_nz> modify_row_end_boundary_after_div_according_to_row_nz_ptr(new modify_row_end_boundary_after_div_according_to_row_nz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    modify_row_end_boundary_after_div_according_to_row_nz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_end_boundary_after_div_according_to_row_nz_ptr));

    // 修改列边界
    shared_ptr<modify_col_start_boundary_after_div_according_to_row_nz> modify_col_start_boundary_after_div_according_to_row_nz_ptr(new modify_col_start_boundary_after_div_according_to_row_nz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    modify_col_start_boundary_after_div_according_to_row_nz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_start_boundary_after_div_according_to_row_nz_ptr));

    shared_ptr<modify_col_end_boundary_after_div_according_to_row_nz> modify_col_end_boundary_after_div_according_to_row_nz_ptr(new modify_col_end_boundary_after_div_according_to_row_nz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    modify_col_end_boundary_after_div_according_to_row_nz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_end_boundary_after_div_according_to_row_nz_ptr));

    // 执行列切分
    shared_ptr<div_col_indices_by_row_nnz> div_col_indices_by_row_nnz_ptr(new div_col_indices_by_row_nnz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    div_col_indices_by_row_nnz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(div_col_indices_by_row_nnz_ptr));

    // 执行值切分
    shared_ptr<div_val_indices_by_row_nnz> div_val_indices_by_row_nnz_ptr(new div_val_indices_by_row_nnz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    div_val_indices_by_row_nnz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(div_val_indices_by_row_nnz_ptr));

    // 执行行切分
    shared_ptr<div_row_indices_by_row_nnz> div_row_indices_by_row_nnz_ptr(new div_row_indices_by_row_nnz(this->meta_data_set_ptr, this->target_matrix_id, this->init_row_size_upper_boundary, this->max_row_size_upper_boundary, this->expansion_rate));
    div_row_indices_by_row_nnz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(div_row_indices_by_row_nnz_ptr));
    if (check)
    {
        max_existing_sub_matrix_id_of_end_row_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "end_row_index");
        max_existing_sub_matrix_id_of_begin_row_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "begin_row_index");
        max_existing_sub_matrix_id_of_begin_col_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "begin_col_index");
        max_existing_sub_matrix_id_of_end_col_index = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "end_col_index");
        max_existing_sub_matrix_id_of_nz_row_indices = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_row_indices");
        max_existing_sub_matrix_id_of_nz_col_indices = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_col_indices");
        max_existing_sub_matrix_id_of_nz_vals = this->meta_data_set_ptr->get_max_sub_matrix_id_of_data_item(GLOBAL_META, "nz_vals");

        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_begin_row_index);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_begin_col_index);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_end_col_index);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_nz_row_indices);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_nz_col_indices);
        assert(max_existing_sub_matrix_id_of_end_row_index == max_existing_sub_matrix_id_of_nz_vals);

        int cur_max_sub_matrix_id = max_existing_sub_matrix_id_of_end_row_index;

        assert(last_max_sub_matrix_id < cur_max_sub_matrix_id);

        for (int cur_sub_matrix_id = last_max_sub_matrix_id + 1; cur_sub_matrix_id <= cur_max_sub_matrix_id; cur_sub_matrix_id++)
        {
            // 三个索引的长度必须吻合
            shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", cur_sub_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", cur_sub_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", cur_sub_matrix_id)->get_metadata_arr();

            assert(row_index_ptr->get_len() == col_index_ptr->get_len());
            assert(row_index_ptr->get_len() == val_ptr->get_len());
        }
    }
    // 再次执行检查

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> row_nz_matrix_div_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->transform_seq.size() == 7);

    // 检查所有的内容
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回对应的操作序列
    return this->transform_seq;
}

string row_nz_matrix_div_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "row_nz_matrix_div_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}