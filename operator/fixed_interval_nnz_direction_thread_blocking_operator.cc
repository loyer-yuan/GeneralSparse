#include "../operator.hpp"

fixed_interval_nnz_direction_thread_blocking_operator::fixed_interval_nnz_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr,
                                                                                                             int target_matrix_id, int nnz_per_BMT, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding)
    : basic_operator("fixed_interval_nnz_direction_thread_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(nnz_per_BMT > 0);

    this->target_matrix_id = target_matrix_id;
    this->nnz_per_BMT = nnz_per_BMT;
    this->row_index_is_relative_to_parent = row_index_is_relative_to_parent;
    this->nz_index_is_relative_to_parent = nz_index_is_relative_to_parent;
    this->nnz_padding = nnz_padding;
}

fixed_interval_nnz_direction_thread_blocking_operator::fixed_interval_nnz_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr,
                                                                                                             int target_matrix_id, int nnz_per_BMT, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_nnz_direction_thread_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{

    assert(target_matrix_id >= 0);
    assert(nnz_per_BMT > 0);

    this->target_matrix_id = target_matrix_id;
    this->nnz_per_BMT = nnz_per_BMT;
    this->row_index_is_relative_to_parent = row_index_is_relative_to_parent;
    this->nz_index_is_relative_to_parent = nz_index_is_relative_to_parent;
    this->nnz_padding = nnz_padding;
}

fixed_interval_nnz_direction_thread_blocking_operator::fixed_interval_nnz_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr,
                                                                                                             int nnz_per_BMT, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_nnz_direction_thread_blocking_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new (this) fixed_interval_nnz_direction_thread_blocking_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), nnz_per_BMT, row_index_is_relative_to_parent, nz_index_is_relative_to_parent, nnz_padding, operator_history);
    this->code_generator_ptr = code_generator_ptr;
}

bool fixed_interval_nnz_direction_thread_blocking_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{

    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool flag = true;
    if (former_operator_implementing.size() == 0)
    {
        if (former_operator_distributing.size() != 0)
        {
            for (int i = 0; i < former_operator_distributing.size(); i++)
            {
                if (former_operator_distributing[i]->get_name().find("fixed_interval_nnz_direction_tblock_blocking_operator") != string::npos || former_operator_distributing[i]->get_name().find("fixed_interval_nnz_direction_warp_blocking_operator") != string::npos)
                {
                    flag = flag & true;
                }
                else
                {
                    flag = flag & false;
                }
            }
        }
        else
        {
            flag = flag & true;
        }
    }

    bool param_flag = true;

    for (int i = 0; i < former_operator_distributing.size(); i++)
    {
        if (former_operator_distributing[i]->get_name().find("fixed_interval_nnz_direction_tblock_blocking_operator") != string::npos)
        {
            shared_ptr<fixed_interval_nnz_direction_tblock_blocking_operator> tblock_level_operator = dynamic_pointer_cast<fixed_interval_nnz_direction_tblock_blocking_operator>(former_operator_distributing[i]);

            if (tblock_level_operator->get_nnz_per_BMTB() % this->nnz_per_BMT != 0)
            {
                param_flag = param_flag & false;
            }
        }

        if (former_operator_distributing[i]->get_name().find("fixed_interval_nnz_direction_warp_blocking_operator") != string::npos)
        {
            shared_ptr<fixed_interval_nnz_direction_warp_blocking_operator> warp_level_operator = dynamic_pointer_cast<fixed_interval_nnz_direction_warp_blocking_operator>(former_operator_distributing[i]);

            if (warp_level_operator->get_nnz_per_BMW() % this->nnz_per_BMT != 0)
            {
                param_flag = param_flag & false;
            }
        }
    }

    if (flag == true && param_flag == true)
    {
        return true;
    }

    return false;
}

bool fixed_interval_nnz_direction_thread_blocking_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->nnz_per_BMT > 0);
    assert(this->target_matrix_id >= 0);

    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    // 检查之前是否存在tblock warp thread级别的切分
    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);

    // 当前内容全部存在
    bool necessary_data = row_indices_existing && col_indices_existing && vals_existing && start_row_boundary && end_row_boundary && (thread_meta_count == 0);

    bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    bool BMTB_nz_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id);

    bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);
    bool BMW_nz_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id);

    assert(BMTB_row_begin_existing ^ BMTB_nz_begin_existing == 0);
    assert(BMW_row_begin_existing ^ BMW_nz_begin_existing == 0);

    // 如果是relative的，那么必须有线程块级别的切分
    if (this->row_index_is_relative_to_parent == true || this->nz_index_is_relative_to_parent)
    {
        necessary_data = necessary_data & (BMTB_row_begin_existing | BMW_nz_begin_existing);
    }

    if (this->nnz_padding == true)
    {
        necessary_data = necessary_data & (~BMTB_row_begin_existing) & (~BMW_row_begin_existing);
    }
    // 检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    if (necessary_data == true && interlance_storage_existing == false)
    {
        return true;
    }

    return false;
}

void fixed_interval_nnz_direction_thread_blocking_operator::run(bool check)
{
    if (check == true)
    {
        assert(this->is_valid_according_to_metadata());
    }
    // 首先做一系列运行前检查，行列值数组长度相等
    if (check == true)
    {
        shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

        assert(row_index_ptr->get_len() == col_index_ptr->get_len());
        assert(col_index_ptr->get_len() == val_ptr->get_len());
    }

    bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);

    if (this->nnz_padding == true)
    {
        shared_ptr<modify_col_indices_by_nnz_pad> modify_col_indices_by_nnz_pad_ptr(new modify_col_indices_by_nnz_pad(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
        modify_col_indices_by_nnz_pad_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(modify_col_indices_by_nnz_pad_ptr));

        shared_ptr<modify_vals_by_nnz_pad> modify_vals_by_nnz_pad_ptr(new modify_vals_by_nnz_pad(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
        modify_vals_by_nnz_pad_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(modify_vals_by_nnz_pad_ptr));

        shared_ptr<modify_row_indices_by_nnz_pad> modify_row_indices_by_nnz_pad_ptr(new modify_row_indices_by_nnz_pad(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
        modify_row_indices_by_nnz_pad_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(modify_row_indices_by_nnz_pad_ptr));
    }

    // 开始nnz切分
    //  thread级别的绝对行索引
    shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction> get_absolute_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
    get_absolute_BMT_start_row_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_row_ptr));

    // 查看相对索引
    if (this->row_index_is_relative_to_parent)
    {
        if (BMW_row_begin_existing == true)
        {
            shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW> get_relative_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
            get_relative_BMT_start_row_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_row_ptr));
        }
        else if (BMTB_row_begin_existing == true)
        {
            shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB> get_relative_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
            get_relative_BMT_start_row_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_row_ptr));
        }
        else
        {
            cout << "fixed_interval_nnz_direction_thread_blocking_operator::run: no parent block" << endl;
            assert(false);
        }
    }

    // thread级别的绝对非零元索引
    shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction> get_absolute_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
    get_absolute_BMT_start_nz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_nz_ptr));

    if (this->nz_index_is_relative_to_parent)
    {
        if (BMW_row_begin_existing == true)
        {
            shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW> get_relative_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
            get_relative_BMT_start_nz_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_nz_ptr));
        }
        else if (BMTB_row_begin_existing == true)
        {
            shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB> get_relative_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_BMT));
            get_relative_BMT_start_nz_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_nz_ptr));
        }
        else
        {
            assert(false);
        }
    }

    if (BMW_row_begin_existing == true)
    {
        shared_ptr<get_begin_BMTs_of_specific_parent_after_blocking> get_BMT_start_ptr(new get_begin_BMTs_of_specific_parent_after_blocking(this->meta_data_set_ptr, WARP_META, this->target_matrix_id));
        get_BMT_start_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_start_ptr));
    }
    else if (BMTB_row_begin_existing == true)
    {
        shared_ptr<get_begin_BMTs_of_specific_parent_after_blocking> get_BMT_start_ptr(new get_begin_BMTs_of_specific_parent_after_blocking(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
        get_BMT_start_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_start_ptr));
    }

    shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_GLOBAL_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, GLOBAL_META, this->target_matrix_id, false));
    get_BMT_size_of_GLOBAL_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_GLOBAL_ptr));
    
    
    this->code_generator_ptr->open_spec_level_of_paral(THREAD_META);
    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> fixed_interval_nnz_direction_thread_blocking_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->nnz_per_BMT > 0);

    // 检查内容中不能空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }
    return this->transform_seq;
}

string fixed_interval_nnz_direction_thread_blocking_operator::convert_to_string()
{
    assert(this->nnz_per_BMT > 0);
    assert(this->target_matrix_id >= 0);
    string return_str = "fixed_interval_nnz_direction_thread_blocking_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",nnz_per_BMT:" + to_string(this->nnz_per_BMT) + "}";

    return return_str;
}