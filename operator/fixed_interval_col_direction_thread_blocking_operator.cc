#include "../operator.hpp"

fixed_interval_col_direction_thread_blocking_operator::fixed_interval_col_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr,
                                                                                                             int target_matrix_id, int fixed_col_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB,
                                                                                                             bool row_index_is_relative_to_BMW, bool nz_index_is_relative_to_BMW, bool is_padding_with_col_size_in_bmt, bool is_col_padding_with_row_max_size_without_empty_row, POS_TYPE padding_pos, vector<shared_ptr<basic_operator>> former_operator)
    : basic_operator("fixed_interval_col_direction_thread_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_col_block_size > 0);
    assert(check_pos_type(padding_pos) == true);
    assert(padding_pos == GLOBAL_META || padding_pos == TBLOCK_META || padding_pos == WARP_META);

    this->target_matrix_id = target_matrix_id;
    this->fixed_col_block_size = fixed_col_block_size;
    this->row_index_is_relative_to_BMTB = row_index_is_relative_to_BMTB;
    this->nz_index_is_relative_to_BMTB = nz_index_is_relative_to_BMTB;
    this->row_index_is_relative_to_BMW = row_index_is_relative_to_BMW;
    this->nz_index_is_relative_to_BMW = nz_index_is_relative_to_BMW;
    this->is_padding_with_col_size_in_bmt = is_padding_with_col_size_in_bmt;
    this->is_col_padding_with_row_max_size_without_empty_row = is_col_padding_with_row_max_size_without_empty_row;
    this->former_operator = former_operator;
    this->padding_pos = padding_pos;
}

fixed_interval_col_direction_thread_blocking_operator::fixed_interval_col_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr,
                                                                                                             int target_matrix_id, int fixed_col_block_size, bool row_index_is_relative, bool nz_index_is_relative, bool is_padding_with_col_size_in_bmt, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_col_direction_thread_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{

    assert(target_matrix_id >= 0);
    assert(fixed_col_block_size > 0);
    assert(check_pos_type(padding_pos) == true);

    this->fixed_col_block_size = fixed_col_block_size;
    // this->row_index_is_relative_to_BMTB = row_index_is_relative_to_BMTB;
    // this->nz_index_is_relative_to_BMTB = nz_index_is_relative_to_BMTB;
    // this->row_index_is_relative_to_BMW = row_index_is_relative_to_BMW;
    // this->nz_index_is_relative_to_BMW = nz_index_is_relative_to_BMW;
    this->is_padding_with_col_size_in_bmt = is_padding_with_col_size_in_bmt;
    this->is_col_padding_with_row_max_size_without_empty_row = is_col_padding_with_row_max_size_without_empty_row;
    this->former_operator = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);

    bool warp_flag = false, tblock_flag = false;
    for (int i = 0; i < this->former_operator.size(); i++)
    {
        if (this->former_operator[i]->get_name().find("warp") != string::npos)
        {
            warp_flag = true;
        }
        else if (this->former_operator[i]->get_name().find("tblock") != string::npos)
        {
            tblock_flag = true;
        }
    }
    if (warp_flag == true)
    {
        this->padding_pos = WARP_META;
        if (row_index_is_relative == true)
        {
            this->row_index_is_relative_to_BMW = true;
        }
        if (nz_index_is_relative == true)
        {
            this->nz_index_is_relative_to_BMW = true;
        }
    }
    else if (tblock_flag == true)
    {
        this->padding_pos = TBLOCK_META;
        if (row_index_is_relative == true)
        {
            this->row_index_is_relative_to_BMTB = true;
        }
        if (nz_index_is_relative == true)
        {
            this->nz_index_is_relative_to_BMTB = true;
        }
    }
    else
    {
        this->padding_pos = GLOBAL_META;
    }

    assert(this->padding_pos == GLOBAL_META || this->padding_pos == TBLOCK_META || this->padding_pos == WARP_META);
}



fixed_interval_col_direction_thread_blocking_operator::fixed_interval_col_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr,
                                                                                                             int fixed_col_block_size, bool row_index_is_relative, bool nz_index_is_relative, bool is_padding_with_col_size_in_bmt, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_col_direction_thread_blocking_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this)fixed_interval_col_direction_thread_blocking_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), fixed_col_block_size, row_index_is_relative, nz_index_is_relative, is_padding_with_col_size_in_bmt, is_col_padding_with_row_max_size_without_empty_row, operator_history);
    this->code_generator_ptr = code_generator_ptr;
}

bool fixed_interval_col_direction_thread_blocking_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{

    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);

    bool col_direction_flag = false;
    bool thread_flag = false;
    bool interlance_flag = false;
    bool balanced_interval_and_max_padding_flag = false;

    if (former_operator_implementing.size() == 0)
    {
        for (int i = 0; i < former_operator_distributing.size(); i++)
        {
            if (former_operator_distributing[i]->get_name().find("thread") != string::npos)
            {
                thread_flag = true;
            }
            if (former_operator_distributing[i]->get_name().find("col") != string::npos)
            {
                col_direction_flag = true;
            }
            if (former_operator_distributing[i]->get_name().find("interlance") != string::npos)
            {
                interlance_flag = true;
            }
            if (former_operator_distributing[i]->get_name().find("balanced_interval") != string::npos)
            {
                balanced_interval_and_max_padding_flag = true;
            }
        }
        balanced_interval_and_max_padding_flag = balanced_interval_and_max_padding_flag & this->is_col_padding_with_row_max_size_without_empty_row;

        if (thread_flag == false && col_direction_flag == false && interlance_flag == false && balanced_interval_and_max_padding_flag == false)
        {
            return true;
        }
    }

    return false;
}

bool fixed_interval_col_direction_thread_blocking_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->fixed_col_block_size > 0);
    assert(this->target_matrix_id >= 0);

    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    //检查之前是否存在bmt级别的切分
    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);

    // 当前内容全部存在
    bool necessary_data = row_indices_existing && col_indices_existing && vals_existing && start_row_boundary && end_row_boundary && (thread_meta_count == 0);
    bool balanced_interval_and_max_padding_flag = false;
    for (int i = 0; i < former_operator.size(); i++)
    {
        if (this->former_operator[i]->get_name().find("balanced_interval") != string::npos)
        {
            balanced_interval_and_max_padding_flag = true;
        }
    }
    balanced_interval_and_max_padding_flag = balanced_interval_and_max_padding_flag & this->is_col_padding_with_row_max_size_without_empty_row;
    bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    bool BMTB_nz_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id);
    bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);
    bool BMW_nz_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id);

    assert(BMTB_row_begin_existing ^ BMTB_nz_begin_existing == 0);
    assert(BMW_row_begin_existing ^ BMW_nz_begin_existing == 0);

    // 如果是relative的，那么必须有线程块级别的切分
    if (this->row_index_is_relative_to_BMTB == true)
    {
        necessary_data = necessary_data & BMTB_row_begin_existing;
    }

    if (this->nz_index_is_relative_to_BMTB == true)
    {
        necessary_data = necessary_data & BMTB_nz_begin_existing;
    }

    if (this->row_index_is_relative_to_BMW == true)
    {
        necessary_data = necessary_data & BMW_row_begin_existing;
    }

    if (this->nz_index_is_relative_to_BMW == true)
    {
        necessary_data = necessary_data & BMW_nz_begin_existing;
    }

    if (this->is_col_padding_with_row_max_size_without_empty_row == true)
    {
        if (this->padding_pos == TBLOCK_META)
        {
            necessary_data = necessary_data & BMTB_row_begin_existing;
        }
        else if (this->padding_pos == WARP_META)
        {
            necessary_data = necessary_data & BMW_row_begin_existing;
        }
        else if (this->padding_pos == GLOBAL_META)
        {
            necessary_data = necessary_data;
        }
        else
        {
            necessary_data = false;
        }
    }
    if (BMTB_row_begin_existing == true)
    {
        if (has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id) == false)
        {
            necessary_data = false;
        }
    }
    if (BMW_row_begin_existing == true)
    {

        if (has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, WARP_META, this->target_matrix_id) == false)
        {
            necessary_data = false;
        }
    }

    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    if (necessary_data == true)
    {
        if (this->is_padding_with_col_size_in_bmt)
        {
            necessary_data = necessary_data & padding_rate_valid_col_direction_with_multiple(this->meta_data_set_ptr, this->fixed_col_block_size, this->target_matrix_id);
        }
        if (this->is_col_padding_with_row_max_size_without_empty_row)
        {
            necessary_data = necessary_data & padding_rate_valid_col_direction_with_max_size_in_parent(this->meta_data_set_ptr, this->fixed_col_block_size, this->padding_pos, this->target_matrix_id);
        }
    }

    if (necessary_data == true && interlance_storage_existing == false && balanced_interval_and_max_padding_flag == false)
    {
        return true;
    }

    return false;
}

void fixed_interval_col_direction_thread_blocking_operator::run(bool check)
{
    if (check == true)
    {
        assert(this->is_valid_according_to_metadata());
        if (this->former_operator.size() > 0)
        {
            bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
            bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);
            assert(BMTB_row_begin_existing == true || BMW_row_begin_existing == true);
        }
        else
        {
            bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
            bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);
            assert(BMTB_row_begin_existing == false && BMW_row_begin_existing == false);
        }
    }

    // 检查之前

    // 首先做一系列运行前检查，行列值数组长度相等

    if (check == true)
    {
        shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

        assert(row_index_ptr->get_len() == col_index_ptr->get_len());
        assert(col_index_ptr->get_len() == val_ptr->get_len());
    }
    //是否经历过BMTB 以及 BMW 级别的分块
    bool BMTB_blocking_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    bool BMW_blocking_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);

    //根据父块中最大行长度进行padding
    if (this->is_col_padding_with_row_max_size_without_empty_row == true)
    {
        shared_ptr<modify_col_indices_by_col_pad_parent_blk_to_max_row_size> col_padding_transform_ptr(new modify_col_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, this->padding_pos, this->target_matrix_id, false));
        col_padding_transform_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

        shared_ptr<modify_vals_by_col_pad_parent_blk_to_max_row_size> val_padding_transform_ptr(new modify_vals_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, this->padding_pos, this->target_matrix_id, false));
        val_padding_transform_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

        shared_ptr<modify_row_indices_by_col_pad_parent_blk_to_max_row_size> row_padding_transform_ptr(new modify_row_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, this->padding_pos, this->target_matrix_id, false));
        row_padding_transform_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));
    }

    //根据指定长度padding
    if (this->is_padding_with_col_size_in_bmt == true)
    {
        shared_ptr<modify_col_indices_by_col_pad_in_sub_matrix> col_padding_transform_ptr(new modify_col_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
        col_padding_transform_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

        shared_ptr<modify_vals_by_col_pad_in_sub_matrix> val_padding_transform_ptr(new modify_vals_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
        val_padding_transform_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

        shared_ptr<modify_row_indices_by_col_pad_in_sub_matrix> row_padding_transform_ptr(new modify_row_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
        row_padding_transform_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));
    }

    // 查看当前是否有BMTB、BMW级别的内容，如果有则需要重新执行之前的operator
    if (BMTB_blocking_existing == true && (is_padding_with_col_size_in_bmt == true || is_col_padding_with_row_max_size_without_empty_row == true))
    {
        vector<string> item_name_tblock = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
        //先删除之前已有的item
        for (int i = 0; i < item_name_tblock.size(); i++)
        {
            shared_ptr<remove_item_of_metadata> remove_tblock_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_tblock[i], TBLOCK_META));
            remove_tblock_item_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(remove_tblock_item_ptr));
        }
    }
    if (BMW_blocking_existing == true && (is_padding_with_col_size_in_bmt == true || is_col_padding_with_row_max_size_without_empty_row == true))
    {
        //先删除之前已有的item
        vector<string> item_name_warp = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
        //先删除之前已有的item
        for (int i = 0; i < item_name_warp.size(); i++)
        {
            shared_ptr<remove_item_of_metadata> remove_warp_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_warp[i], WARP_META));
            remove_warp_item_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(remove_warp_item_ptr));
        }
    }

    //重新执行前序operator
    if (is_padding_with_col_size_in_bmt == true || is_col_padding_with_row_max_size_without_empty_row == true)
    {
        for (int i = 0; i < former_operator.size(); i++)
        {
            shared_ptr<basic_operator> rerun_operator = former_operator[i];
            int prev_size = rerun_operator->get_data_transform_sequence().size();
            rerun_operator->set_padding_to_false();
            rerun_operator->run(check);
            //重执行的记录更新，只记录新执行的
            for (int j = prev_size; j < rerun_operator->get_data_transform_sequence().size(); j++)
            {
                this->set_transform_seq(rerun_operator->get_data_transform_sequence()[j]);
            }
        }
    }

    //开始列切分
    // thread级别的绝对行索引
    shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction> get_absolute_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
    get_absolute_BMT_start_row_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_row_ptr));

    // thread级别的绝对非零元索引
    shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction> get_absolute_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
    get_absolute_BMT_start_nz_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_nz_ptr));

    // thread级别相对行索引：BMTB
    if (this->row_index_is_relative_to_BMTB == true)
    {
        //简单检查tblock数据
        if (check)
        {
            shared_ptr<universal_array> tblock_first_row_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> tblock_first_nz_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
            assert(tblock_first_row_ptr->get_len() == tblock_first_nz_ptr->get_len());
        }

        shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMTB> get_relative_to_BMBT_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
        get_relative_to_BMBT_BMT_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_to_BMBT_BMT_start_row_ptr));
    }

    // thread级别的相对非零元索引：BMTB
    if (this->nz_index_is_relative_to_BMTB == true)
    {
        //简单检查tblock数据
        if (check)
        {
            shared_ptr<universal_array> tblock_first_row_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> tblock_first_nz_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
            assert(tblock_first_row_ptr->get_len() == tblock_first_nz_ptr->get_len());
        }

        // 获得相对非零元索引
        shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents> get_relative_to_BMTB_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size, TBLOCK_META));
        get_relative_to_BMTB_BMT_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_to_BMTB_BMT_start_nz_ptr));
    }

    // bmt在tblock中的偏移
    if (BMTB_blocking_existing == true)
    {
        shared_ptr<get_begin_BMTs_of_specific_parent_after_blocking> get_begin_BMTs_of_BMTB_ptr(new get_begin_BMTs_of_specific_parent_after_blocking(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
        get_begin_BMTs_of_BMTB_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMTs_of_BMTB_ptr));
    }

    // bmt size 在tblock之中
    if (is_padding_with_col_size_in_bmt == true && BMTB_blocking_existing == true) //|| is_col_padding_with_row_max_size_without_empty_row == true) && BMW_blocking_existing == false)
    {
        shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_BMTB_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id, false));
        get_BMT_size_of_BMTB_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_BMTB_ptr));
    }

    // thread级别相对行索引：BMW
    if (this->row_index_is_relative_to_BMW == true)
    {
        //简单检查warp数据
        if (check)
        {
            shared_ptr<universal_array> warp_first_row_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> warp_first_nz_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
            assert(warp_first_row_ptr->get_len() == warp_first_nz_ptr->get_len());
        }

        shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW> get_relative_to_BMW_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size));
        get_relative_to_BMW_BMT_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_to_BMW_BMT_start_row_ptr));
    }

    // thread级别的相对非零元索引：BMW
    if (this->nz_index_is_relative_to_BMW == true)
    {
        //简单检查warp数据
        if (check)
        {
            shared_ptr<universal_array> warp_first_row_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> warp_first_nz_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
            assert(warp_first_row_ptr->get_len() == warp_first_nz_ptr->get_len());
        }

        shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents> get_relative_to_BMW_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_col_block_size, WARP_META));
        get_relative_to_BMW_BMT_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_to_BMW_BMT_start_nz_ptr));
    }

    if (BMW_blocking_existing == true)
    {
        // bmt在warp中的偏移
        shared_ptr<get_begin_BMTs_of_specific_parent_after_blocking> get_begin_BMTs_of_BMW_ptr(new get_begin_BMTs_of_specific_parent_after_blocking(this->meta_data_set_ptr, WARP_META, this->target_matrix_id));
        get_begin_BMTs_of_BMW_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMTs_of_BMW_ptr));
    }

    // bmt size 在warp之中
    if (is_padding_with_col_size_in_bmt == true && BMW_blocking_existing == true) // || is_col_padding_with_row_max_size_without_empty_row == true)
    {
        shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_BMW_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, WARP_META, this->target_matrix_id, false));
        get_BMT_size_of_BMW_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_BMW_ptr));
    }

    if (is_padding_with_col_size_in_bmt == true)
    {
        shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_GLOBAL_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, GLOBAL_META, this->target_matrix_id, false));
        get_BMT_size_of_GLOBAL_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_GLOBAL_ptr));
    }
    this->code_generator_ptr->open_spec_level_of_paral(THREAD_META);

    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> fixed_interval_col_direction_thread_blocking_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->fixed_col_block_size > 0);

    // 检查内容中不能空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }
    return this->transform_seq;
}

string fixed_interval_col_direction_thread_blocking_operator::convert_to_string()
{
    assert(this->fixed_col_block_size > 0);
    assert(this->target_matrix_id >= 0);
    string return_str = "fixed_interval_col_direction_thread_blocking_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_col_block_size:" + to_string(this->fixed_col_block_size) + ",is_padding:" + to_string(this->is_padding_with_col_size_in_bmt || this->is_col_padding_with_row_max_size_without_empty_row) + "}";

    return return_str;
}