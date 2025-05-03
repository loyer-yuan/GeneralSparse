#include "../operator.hpp"

col_based_sort_operator::col_based_sort_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_operator("col_based_sort_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
}

col_based_sort_operator::col_based_sort_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history)
    : basic_operator("col_based_sort_operator", meta_data_set_ptr, CONVERTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
}


col_based_sort_operator::col_based_sort_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history)
    : basic_operator("col_based_sort_operator", code_generator_ptr->get_metadata_set(), CONVERTING_OP, code_generator_ptr->get_sub_matrix_id())
{

}

bool col_based_sort_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
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
                if (former_operator_converting[i]->get_name().find("sort") != string::npos)
                {
                    if(former_operator_converting[i]->get_target_matrix_id() == this->get_target_matrix_id())
                    {
                        self_flag = true;
                    }
                }
            }
            if (self_flag == false)
            {
                return true;
            }
        }
    }
    return false;
}

bool col_based_sort_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());

    // 只有当存在行、列、值三个数组，并且不存在origin row index的时候才能使用
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    // 暂时先认为只有sort才改变行排序
    bool origin_row_index_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id);

    if (row_indices_existing == true && col_indices_existing == true && vals_existing == true && origin_row_index_existing == false)
    {
        return true;
    }

    return false;
}

// 执行具体的排序操作
void col_based_sort_operator::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->is_valid_according_to_metadata());
    }

    shared_ptr<get_row_order_by_col> get_row_order_by_col_transform(new get_row_order_by_col(this->meta_data_set_ptr, this->target_matrix_id));
    get_row_order_by_col_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_row_order_by_col_transform));

    // 执行值的重排
    shared_ptr<reorder_val_by_index> reorder_val_by_index_transform(new reorder_val_by_index(this->meta_data_set_ptr, this->target_matrix_id));
    reorder_val_by_index_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(reorder_val_by_index_transform));

    // 执行列的重排
    shared_ptr<reorder_col_by_index> reorder_col_by_index_transform(new reorder_col_by_index(this->meta_data_set_ptr, this->target_matrix_id));
    reorder_col_by_index_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(reorder_col_by_index_transform));

    // 执行行的重排
    shared_ptr<reorder_row_by_index> reorder_row_by_index_transform(new reorder_row_by_index(this->meta_data_set_ptr, this->target_matrix_id));
    reorder_row_by_index_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(reorder_row_by_index_transform));

    // 如果存在空行就处理结尾的空行
    shared_ptr<remove_empty_row_in_end_of_sub_matrix> remove_empty_row_transform(new remove_empty_row_in_end_of_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id));
    remove_empty_row_transform->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(remove_empty_row_transform));

    this->is_run = true;
}

// 给出当前op的操作序列，用以生成format conversion
vector<shared_ptr<transform_step_record_item>> col_based_sort_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回所有的操作集合
    return this->transform_seq;
}

string col_based_sort_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "col_based_sort_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}