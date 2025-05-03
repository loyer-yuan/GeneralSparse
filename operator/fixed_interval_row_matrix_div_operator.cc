#include "../operator.hpp"

// 初始化
fixed_interval_row_matrix_div_operator::fixed_interval_row_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_interval_size)
    : basic_operator("fixed_interval_row_matrix_div_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_interval_size > 0);

    this->fixed_row_interval_size = fixed_row_interval_size;
}

fixed_interval_row_matrix_div_operator::fixed_interval_row_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_interval_size, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_row_matrix_div_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_interval_size > 0);

    this->fixed_row_interval_size = fixed_row_interval_size;
}



fixed_interval_row_matrix_div_operator::fixed_interval_row_matrix_div_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_interval_size, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_row_matrix_div_operator", code_generator_ptr->get_metadata_set(), CONVERTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this)fixed_interval_row_matrix_div_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), fixed_row_interval_size, operator_history);
}

bool fixed_interval_row_matrix_div_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
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
    if(self_flag == false)
    {
        return true;
    }

    return false;
}
bool fixed_interval_row_matrix_div_operator::is_valid_according_to_metadata()
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

    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 查看当前行的数量
    unsigned long row_num = end_row_index - begin_row_index + 1;

    // 查看当前的行数量要分多少个子块
    unsigned long sub_matrix_num = row_num / this->get_fixed_row_interval_size();

    // 如果不能整除，那么就需要多一个子块
    if (row_num % this->get_fixed_row_interval_size() != 0)
    {
        sub_matrix_num = sub_matrix_num + 1;
    }

    // 读出行索引，查看空白行
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 用一个数组记录空桶
    vector<bool> empty_bin_flag(sub_matrix_num);

    for (unsigned long i = 0; i < sub_matrix_num; i++)
    {
        empty_bin_flag[i] = true;
    }

    // 遍历所有行数据
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i++)
    {
        unsigned long cur_row_index = nz_row_indices_ptr->read_integer_from_arr(i);

        unsigned long cur_sub_matrix_id = cur_row_index / this->get_fixed_row_interval_size();
        assert(cur_sub_matrix_id < sub_matrix_num);
        empty_bin_flag[cur_sub_matrix_id] = false;
    }

    int non_empty_matrix_num = 0;
    // 遍历所有桶，如果不是空桶，那就使用
    for (unsigned long i = 0; i < empty_bin_flag.size(); i++)
    {
        // 如果当前不是空块
        if (empty_bin_flag[i] == false)
        {
            non_empty_matrix_num += 1;
        }
    }

    // 上述内容全部存在才能执行
    if (start_row_boundary == true && end_row_boundary == true && start_col_boundary == true && end_col_boundary == true && interlance_storage_existing == false)
    {
        if (row_indices_existing == true && col_indices_existing == true && vals_existing == true)
        {
            // 查看被切的块在行方向的宽度，宽度不能低于切分的宽度
            // 读出来当前子块的首行索引和最后一行的索引
            unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
            unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

            // 查看当前行的数量
            unsigned long row_num = end_row_index - begin_row_index + 1;

            // 分块的宽度要小于矩阵的行数量
            if (row_num > this->get_fixed_row_interval_size() && non_empty_matrix_num <= get_config()["MAX_DIV_TIMES_OF_DIV"].as_integer())
            {
                return true;
            }
        }
    }

    return false;
}

void fixed_interval_row_matrix_div_operator::run(bool check)
{
    if (check)
    {
        assert(this->is_valid_according_to_metadata());
        // cout << "fixed_interval_row_matrix_div_operator::run: is checked" << endl;
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
    shared_ptr<modify_row_start_boundary_after_fixed_div_in_row_direction> modify_row_start_boundary_after_fixed_div_in_row_direction_ptr(new modify_row_start_boundary_after_fixed_div_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    // 执行
    modify_row_start_boundary_after_fixed_div_in_row_direction_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_start_boundary_after_fixed_div_in_row_direction_ptr));

    shared_ptr<modify_row_end_boundary_after_fixed_div_in_row_direction> modify_row_end_boundary_after_fixed_div_in_row_direction_ptr(new modify_row_end_boundary_after_fixed_div_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    modify_row_end_boundary_after_fixed_div_in_row_direction_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_end_boundary_after_fixed_div_in_row_direction_ptr));

    // 修改列边界
    shared_ptr<modify_col_start_boundary_after_fixed_div_in_row_direction> modify_col_start_boundary_after_fixed_div_in_row_direction_ptr(new modify_col_start_boundary_after_fixed_div_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    modify_col_start_boundary_after_fixed_div_in_row_direction_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_start_boundary_after_fixed_div_in_row_direction_ptr));

    shared_ptr<modify_col_end_boundary_after_fixed_div_in_row_direction> modify_col_end_boundary_after_fixed_div_in_row_direction_ptr(new modify_col_end_boundary_after_fixed_div_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    modify_col_end_boundary_after_fixed_div_in_row_direction_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_end_boundary_after_fixed_div_in_row_direction_ptr));

    // 执行列切分
    shared_ptr<fixed_div_col_indices_by_corr_row_indices> fixed_div_col_indices_by_corr_row_indices_ptr(new fixed_div_col_indices_by_corr_row_indices(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    fixed_div_col_indices_by_corr_row_indices_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(fixed_div_col_indices_by_corr_row_indices_ptr));

    // 执行值切分
    shared_ptr<fixed_div_vals_by_corr_row_indices> fixed_div_vals_by_corr_row_indices_ptr(new fixed_div_vals_by_corr_row_indices(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    fixed_div_vals_by_corr_row_indices_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(fixed_div_vals_by_corr_row_indices_ptr));

    // 执行行切分
    shared_ptr<fixed_div_row_indices> fixed_div_row_indices_ptr(new fixed_div_row_indices(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_interval_size));
    fixed_div_row_indices_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(fixed_div_row_indices_ptr));
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

            // 取出行索引的最后一位，满足切分的范围
            unsigned long last_row_index = row_index_ptr->read_integer_from_arr(row_index_ptr->get_len() - 1);
            assert(last_row_index < this->fixed_row_interval_size);
        }
    }
    // 再次执行检查

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> fixed_interval_row_matrix_div_operator::get_data_transform_sequence()
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

string fixed_interval_row_matrix_div_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "fixed_interval_row_matrix_div_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_interval_size:" + to_string(this->fixed_row_interval_size) + "}";

    return return_str;
}