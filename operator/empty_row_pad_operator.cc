#include "../operator.hpp"

empty_row_pad_operator::empty_row_pad_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_operator("empty_row_pad_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
}

empty_row_pad_operator::empty_row_pad_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history)
    : basic_operator("empty_row_pad_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
}

empty_row_pad_operator::empty_row_pad_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history)
    : basic_operator("empty_row_pad_operator", code_generator_ptr->get_metadata_set(), CONVERTING_OP, code_generator_ptr->get_sub_matrix_id())
{
}

bool empty_row_pad_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_converting = operator_history->read_operator_context_arr(CONVERTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool sort_flag = false;
    // if (former_operator_implementing.size() == 0)
    // {
    //     if (former_operator_distributing.size() == 0)
    //     {
    //         // for (int i = 0; i < former_operator_converting.size(); i++)
    //         // {
    //         //     if (former_operator_converting[i]->get_name().find("sort") != string::npos)
    //         //     {
    //         //         sort_flag = true;
    //         //     }
    //         // }
    //         // if(sort_flag == false)
    //         // {
    //         //     return true;
    //         // }
    //         if (former_operator_converting.size() == 0)
    //         {
    //             return true;
    //         }
    //     }
    // }
    bool self_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        if (former_operator_distributing.size() == 0)
        {
            for (int i = 0; i < former_operator_converting.size(); i++)
            {
                if (former_operator_converting[i]->get_name().find("sort_operator") != string::npos)
                {
                    if (former_operator_converting[i]->get_target_matrix_id() == this->get_target_matrix_id())
                    {
                        sort_flag = true;
                    }
                }
                if (former_operator_converting[i]->get_name().find("empty_row_pad_operator") != string::npos)
                {
                    if (former_operator_converting[i]->get_target_matrix_id() == this->get_target_matrix_id())
                    {
                        self_flag = true;
                    }
                }
            }
            if (self_flag == false && sort_flag == false)
            {
                return true;
            }
        }
    }
    return false;
}

bool empty_row_pad_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());

    // 只有当存在行、列、值三个数组存在
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    // TODO：空行检查
    shared_ptr<universal_array> row_index_arr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    unsigned long end_row = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long begin_row = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    unsigned long real_max_row_index = row_index_arr->read_integer_from_arr(row_index_arr->get_len() - 1);

    assert(end_row >= begin_row);

    unsigned long max_logic_relative_row_index = end_row - begin_row;

    if (real_max_row_index > max_logic_relative_row_index)
    {
        end_row = begin_row + real_max_row_index;
    }
    unsigned long relative_max_row_index = end_row - begin_row;

    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(row_index_arr, 0, relative_max_row_index, 0, row_index_arr->get_len() - 1);

    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);
    int warp_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
    int tblock_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
    bool flag = (thread_meta_count == 0) & (warp_meta_count == 0) & (tblock_meta_count == 0); 
    if (row_indices_existing == true && col_indices_existing == true && vals_existing == true && start_row_boundary == true && end_row_boundary == true && interlance_storage_existing == false && flag == true)
    {
        if (find(nnz_of_each_row.begin(), nnz_of_each_row.end(), 0) != nnz_of_each_row.end())
        {
            if (padding_rate_valid_empty_padding(this->meta_data_set_ptr, this->target_matrix_id) == true)
            {
                return true;
            }
        }
    }

    return false;
}

// 执行具体的排序操作
void empty_row_pad_operator::run(bool check)
{
    if (check == true)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        // 查看当前能不能执行，之前要通过这个检查，不能通过那就直接退出
        assert(this->is_valid_according_to_metadata());
    }

    // 首先pad 列
    shared_ptr<modify_col_indices_by_empty_pad_in_submatrix> modify_col_indices_by_empty_pad_in_submatrix_transform(new modify_col_indices_by_empty_pad_in_submatrix(this->meta_data_set_ptr, this->target_matrix_id));
    // 然后执行对应的data_transform
    modify_col_indices_by_empty_pad_in_submatrix_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_indices_by_empty_pad_in_submatrix_transform));

    // pad 值
    shared_ptr<modify_vals_by_empty_pad_in_submatrix> modify_vals_by_empty_pad_in_submatrix_transform(new modify_vals_by_empty_pad_in_submatrix(this->meta_data_set_ptr, this->target_matrix_id));
    // 然后执行对应的data_transform
    modify_vals_by_empty_pad_in_submatrix_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_vals_by_empty_pad_in_submatrix_transform));

    // pad 行
    shared_ptr<modify_row_indices_by_empty_pad_in_submatrix> modify_row_indices_by_empty_pad_in_submatrix_transform(new modify_row_indices_by_empty_pad_in_submatrix(this->meta_data_set_ptr, this->target_matrix_id));
    // 然后执行对应的data_transform
    modify_row_indices_by_empty_pad_in_submatrix_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_indices_by_empty_pad_in_submatrix_transform));

    this->is_run = true;
}

// 给出当前op的操作序列，用以生成format conversion
vector<shared_ptr<transform_step_record_item>> empty_row_pad_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回所有的操作集合
    return this->transform_seq;
}

string empty_row_pad_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "empty_row_pad_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}