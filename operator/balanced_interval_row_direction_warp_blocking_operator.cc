#include "../operator.hpp"

balanced_interval_row_direction_warp_blocking_operator::balanced_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB)
    : basic_operator("balanced_interval_row_direction_warp_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(nnz_per_interval > 0);


    this->nnz_per_interval = nnz_per_interval;
    this->row_index_is_relative_to_BMTB = row_index_is_relative_to_BMTB;
    this->nz_index_is_relative_to_BMTB = nz_index_is_relative_to_BMTB;

}

balanced_interval_row_direction_warp_blocking_operator::balanced_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, bool row_index_is_relative, bool nz_index_is_relative,  shared_ptr<operator_context> operator_history)
    : basic_operator("balanced_interval_row_direction_warp_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(nnz_per_interval > 0);
    this->nnz_per_interval = nnz_per_interval;
    this->row_index_is_relative_to_BMTB = row_index_is_relative;
    this->nz_index_is_relative_to_BMTB = nz_index_is_relative;
}

balanced_interval_row_direction_warp_blocking_operator::balanced_interval_row_direction_warp_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_interval, bool row_index_is_relative, bool nz_index_is_relative,  shared_ptr<operator_context> operator_history)
    : basic_operator("balanced_interval_row_direction_warp_blocking_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this) balanced_interval_row_direction_warp_blocking_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), nnz_per_interval, row_index_is_relative, nz_index_is_relative,  operator_history);
    this->code_generator_ptr = code_generator_ptr;
}


bool balanced_interval_row_direction_warp_blocking_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{

    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);

    bool col_direction_flag = false;
    bool thread_flag = false;
    bool interlance_flag = false;
    bool warp_flag = false;
    bool nnz_direction_flag = false;
    // 用op的名字来判断对应类型的操作是不是存在
    if (former_operator_implementing.size() == 0)
    {
        for (int i = 0; i < former_operator_distributing.size(); i++)
        {
            // thread的切分是不是存在
            if (former_operator_distributing[i]->get_name().find("thread") != string::npos)
            {
                thread_flag = true;
            }
            // warp的切分是不是存在
            if(former_operator_distributing[i]->get_name().find("warp") != string::npos)
            {
                warp_flag = true;
            }
            // nnz方向的切分不存在
            if(former_operator_distributing[i]->get_name().find("nnz_direction") != string::npos)
            {
                nnz_direction_flag = true;
            }
            // 列切分是不是存在
            if (former_operator_distributing[i]->get_name().find("col") != string::npos)
            {
                col_direction_flag = true;
            }
            // 交错存储是不是存在
            if (former_operator_distributing[i]->get_name().find("interlance") != string::npos)
            {
                interlance_flag = true;
            }
        }
        if (thread_flag == false && warp_flag == false && col_direction_flag == false && interlance_flag == false && nnz_direction_flag == false)
        {
            return true;
        }
    }

    return false;
}

bool balanced_interval_row_direction_warp_blocking_operator::is_valid_according_to_metadata()
{

    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->nnz_per_interval > 0);
    assert(this->target_matrix_id >= 0);

    //检查数据是否存在
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    //之前没有warp、thread级别的切分
    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);
    int warp_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
    bool necessary_data = row_indices_existing && col_indices_existing && vals_existing && start_row_boundary && end_row_boundary && (thread_meta_count == 0) && (warp_meta_count == 0);

    bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    bool BMTB_nz_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id);

    // row_indices 和 nz_indices 只能同时存在或不存在
    // 异或
    assert(BMTB_row_begin_existing ^ BMTB_nz_begin_existing == 0);

    // 如果是relative的，那么必须有线程块级别的切分
    if (this->row_index_is_relative_to_BMTB == true)
    {
        necessary_data = necessary_data & BMTB_row_begin_existing;
    }

    if (this->nz_index_is_relative_to_BMTB == true)
    {
        necessary_data = necessary_data & BMTB_nz_begin_existing;
    }


    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    // 如果之前存在BMTB的切分，那么必须是行切分
    bool valid_BMTB_blocking = false;
    if (BMTB_row_begin_existing == true)
    {
        valid_BMTB_blocking = has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id);
        necessary_data = necessary_data & valid_BMTB_blocking;
    }

    // 当前内容全部存在
    if (necessary_data == true && interlance_storage_existing == false)
    {
        return true;
    }

    return false;
}

// 执行
void balanced_interval_row_direction_warp_blocking_operator::run(bool check)
{
    if (check)
    {
        assert(this->is_valid_according_to_metadata());
        // 首先做一系列运行前检查，行列值数组长度相等
        shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

        assert(row_index_ptr->get_len() == col_index_ptr->get_len());
        assert(col_index_ptr->get_len() == val_ptr->get_len());
    }

    bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);

    if (BMTB_row_begin_existing == true)
    {
        if (check)
        {
            shared_ptr<universal_array> tblock_first_row_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> tblock_first_nz_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

            assert(tblock_first_row_ptr->get_len() == tblock_first_nz_ptr->get_len());
        }

        shared_ptr<get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB> get_absolute_BMW_start_row_ptr(new get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_interval));
        get_absolute_BMW_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_row_ptr));

        //首行相对索引
        if (this->row_index_is_relative_to_BMTB == true)
        {
            // 获得相对行索引
            shared_ptr<get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB> get_relative_BMW_start_row_ptr(new get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_interval));
            get_relative_BMW_start_row_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMW_start_row_ptr));
        }

        shared_ptr<get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB> get_absolute_BMW_start_nz_ptr(new get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_interval));
        get_absolute_BMW_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_nz_ptr));

        // 相对非零元索引
        if (this->nz_index_is_relative_to_BMTB == true)
        {
            // 获得相对非零元索引
            shared_ptr<get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB> get_relative_BMW_start_nz_ptr(new get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_interval));
            get_relative_BMW_start_nz_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMW_start_nz_ptr));
        }

        // 每一个BMTB中BMW的相对索引
        shared_ptr<get_begin_BMWs_of_BMTB_after_blocking_in_row_direction> get_begin_BMW_of_BMTB_ptr(new get_begin_BMWs_of_BMTB_after_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id));
        get_begin_BMW_of_BMTB_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMW_of_BMTB_ptr));
    }
    else
    {
        assert(this->row_index_is_relative_to_BMTB == false);
        assert(this->nz_index_is_relative_to_BMTB == false);
        // 两个，row和nz绝对索引，并且这里不可能出现相对索引
        shared_ptr<get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction> get_absolute_BMW_start_row_ptr(new get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_interval));
        get_absolute_BMW_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_row_ptr));

        shared_ptr<get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction> get_absolute_BMW_start_nz_ptr(new get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->nnz_per_interval));
        get_absolute_BMW_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_nz_ptr));

    }
    if (check)
    {
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id));
    }
    this->code_generator_ptr->open_spec_level_of_paral(WARP_META);

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> balanced_interval_row_direction_warp_blocking_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->nnz_per_interval > 0);

    // 检查内容中不能空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回对应序列
    return this->transform_seq;
}
string balanced_interval_row_direction_warp_blocking_operator::convert_to_string()
{
    assert(this->nnz_per_interval > 0);
    assert(this->target_matrix_id >= 0);

    string return_str = "balanced_interval_row_direction_warp_blocking_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",nnz_per_interval:" + to_string(this->nnz_per_interval)  + "}";

    return return_str;
}
