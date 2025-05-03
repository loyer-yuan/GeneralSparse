#include "../operator.hpp"

interlance_storage_operator::interlance_storage_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, POS_TYPE pos)
    : basic_operator("interlance_storage_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(pos == GLOBAL_META || pos == TBLOCK_META || pos == WARP_META);

    this->pos = pos;
}

interlance_storage_operator::interlance_storage_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history)
    : basic_operator("interlance_storage_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    vector<shared_ptr<basic_operator>> former_operator = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);

    bool warp_flag = false, tblock_flag = false;

    for (int i = 0; i < former_operator.size(); i++)
    {
        if (former_operator[i]->get_name().find("warp") != string::npos)
        {
            warp_flag = true;
        }
        else if (former_operator[i]->get_name().find("tblock") != string::npos)
        {
            tblock_flag = true;
        }
    }
    if (warp_flag == true)
    {
        this->pos = WARP_META;
    }
    else if (tblock_flag == true)
    {
        this->pos = TBLOCK_META;
    }
    else
    {
        this->pos = GLOBAL_META;
    }

    assert(this->pos == GLOBAL_META || this->pos == TBLOCK_META || this->pos == WARP_META);
}


interlance_storage_operator::interlance_storage_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history)
    : basic_operator("interlance_storage_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    code_generator_ptr->set_interleave_storage();
    new(this)interlance_storage_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), operator_history);
}


bool interlance_storage_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);

    bool thread_col_flag_with_multiple_padding = false;
    bool thread_row_flag_with_col_padding = false;
    bool interlance_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        for (int i = 0; i < former_operator_distributing.size(); i++)
        {
            if (former_operator_distributing[i]->get_name().find("fixed_interval_col_direction_thread_blocking_operator") != string::npos)
            {
                if (former_operator_distributing[i]->get_is_padding_with_col_size_in_bmt() == true)
                {
                    thread_col_flag_with_multiple_padding = true;
                }
            }
            if (former_operator_distributing[i]->get_name().find("fixed_interval_row_direction_thread_blocking_operator") != string::npos)
            {
                if (former_operator_distributing[i]->get_is_col_padding_with_row_max_size_with_empty_row() == true)
                {
                    thread_row_flag_with_col_padding = true;
                }
            }
            if (former_operator_distributing[i]->get_name().find("interlance") != string::npos)
            {
                interlance_flag = true;
            }
        }
        if (interlance_flag == false)
        {
            if (thread_col_flag_with_multiple_padding == true || thread_row_flag_with_col_padding == true)
            {
                return true;
            }
        }
    }

    return false;
}

bool interlance_storage_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->target_matrix_id >= 0);
    assert(this->pos == GLOBAL_META || this->pos == TBLOCK_META || this->pos == WARP_META);

    //检查是否存在交错存储
    bool interlance_storage_existing = false;

    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    //检查要交错存储的数据是否存在
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);

    bool parent_bmt_existing = false;
    //交错存储的父块内容是否存在
    if (this->pos != GLOBAL_META)
    {
        parent_bmt_existing = this->meta_data_set_ptr->is_exist(this->pos, "first_BMT_indices", this->target_matrix_id);
    }
    else
    {
        parent_bmt_existing = true;
    }
    // bool has_same_size = same_BMT_size_in_parent(this->meta_data_set_ptr, this->pos, this->target_matrix_id);
    bool has_same_size = this->meta_data_set_ptr->is_exist(this->pos, "BMT_size_of_each_blk", this->target_matrix_id);

    bool neccessary_data = row_indices_existing & col_indices_existing & vals_existing & parent_bmt_existing & has_same_size;

    if (neccessary_data == true && interlance_storage_existing == false)
    {
        return true;
    }

    return false;
}

void interlance_storage_operator::run(bool check)
{
    if (check)
    {
        assert(this->is_valid_according_to_metadata());

        assert(this->meta_data_set_ptr->is_exist(this->pos, "BMT_size_of_each_blk", this->target_matrix_id) == true);
    }

    // if (BMT_size_parent_existing == false)
    // {
    //     //计算BMT_size
    //     shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_each_parent_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, this->pos, this->target_matrix_id));
    //     get_BMT_size_of_each_parent_ptr->run();
    //     this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_each_parent_ptr));
    // }

    // 开始列、值、行的交错存储
    shared_ptr<modify_col_indices_by_interlance_storage> modify_col_indices_by_interlance_storage_ptr(new modify_col_indices_by_interlance_storage(this->meta_data_set_ptr, this->pos, this->target_matrix_id));
    modify_col_indices_by_interlance_storage_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_indices_by_interlance_storage_ptr));

    shared_ptr<modify_vals_by_interlance_storage> modify_vals_by_interlance_storage_ptr(new modify_vals_by_interlance_storage(this->meta_data_set_ptr, this->pos, this->target_matrix_id));
    modify_vals_by_interlance_storage_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_vals_by_interlance_storage_ptr));

    shared_ptr<modify_row_indices_by_interlance_storage> modify_row_indices_by_interlance_storage_ptr(new modify_row_indices_by_interlance_storage(this->meta_data_set_ptr, this->pos, this->target_matrix_id));
    modify_row_indices_by_interlance_storage_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_indices_by_interlance_storage_ptr));

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> interlance_storage_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(pos == GLOBAL_META || pos == TBLOCK_META || pos == WARP_META);

    assert(this->is_run == true);

    return this->transform_seq;
}

string interlance_storage_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(pos == GLOBAL_META || pos == TBLOCK_META || pos == WARP_META);

    string return_str = "interlance_storage_operator::{name:\"" + this->name + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}