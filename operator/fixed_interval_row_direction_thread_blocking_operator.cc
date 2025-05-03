#include "../operator.hpp"

fixed_interval_row_direction_thread_blocking_operator::fixed_interval_row_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool is_row_padding, bool is_col_padding_with_row_max_size_with_empty_row, bool is_col_padding_with_col_size, int col_size, vector<shared_ptr<basic_operator>> former_operator)
    : basic_operator("fixed_interval_row_direction_thread_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    if (is_row_padding == true)
    {
        assert(row_index_is_relative_to_parent == false);
        assert(nz_index_is_relative_to_parent == false);
    }

    this->fixed_row_block_size = fixed_row_block_size;
    this->row_index_is_relative_to_parent = row_index_is_relative_to_parent;
    this->nz_index_is_relative_to_parent = nz_index_is_relative_to_parent;
    this->is_row_padding = is_row_padding;
    this->former_operator = former_operator;
    this->is_col_padding_with_row_max_size_with_empty_row = is_col_padding_with_row_max_size_with_empty_row;
    this->is_col_padding_with_col_size = is_col_padding_with_col_size;
    this->col_size = col_size;
}

fixed_interval_row_direction_thread_blocking_operator::fixed_interval_row_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative, bool nz_index_is_relative, bool is_row_padding, bool is_col_padding_with_row_max_size_with_empty_row, bool is_col_padding_with_col_size, int col_size, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_row_direction_thread_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);
    if (is_row_padding == true)
    {
        assert(row_index_is_relative == false);
        assert(nz_index_is_relative == false);
    }

    this->fixed_row_block_size = fixed_row_block_size;
    this->row_index_is_relative_to_parent = row_index_is_relative;
    this->nz_index_is_relative_to_parent = nz_index_is_relative;
    this->is_row_padding = is_row_padding;
    this->is_col_padding_with_row_max_size_with_empty_row = is_col_padding_with_row_max_size_with_empty_row;
    this->former_operator = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    this->is_col_padding_with_col_size = is_col_padding_with_col_size;
    this->col_size = col_size;
}

fixed_interval_row_direction_thread_blocking_operator::fixed_interval_row_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_block_size, bool row_index_is_relative, bool nz_index_is_relative, bool is_row_padding, bool is_col_padding_with_row_max_size_with_empty_row, bool is_col_padding_with_col_size, int col_size, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_row_direction_thread_blocking_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new (this) fixed_interval_row_direction_thread_blocking_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), fixed_row_block_size, row_index_is_relative, nz_index_is_relative, is_row_padding, is_col_padding_with_row_max_size_with_empty_row, is_col_padding_with_col_size, col_size, operator_history);
    this->code_generator_ptr = code_generator_ptr;
}

bool fixed_interval_row_direction_thread_blocking_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
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
        balanced_interval_and_max_padding_flag = balanced_interval_and_max_padding_flag & is_col_padding_with_row_max_size_with_empty_row;
        if (thread_flag == false && col_direction_flag == false && interlance_flag == false && balanced_interval_and_max_padding_flag == false)
        {
            return true;
        }
    }

    return false;
}

bool fixed_interval_row_direction_thread_blocking_operator::is_valid_according_to_metadata()
{

    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->fixed_row_block_size > 0);
    assert(this->target_matrix_id >= 0);

    // 如果padding，则之前不存在BMTB以及BMW的分块
    if (is_row_padding == true)
    {
        assert(row_index_is_relative_to_parent == false);
        assert(nz_index_is_relative_to_parent == false);
    }

    // 检查数据是否存在
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    // 之前没有thread级别的切分
    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);

    bool necessary_data = row_indices_existing && col_indices_existing && vals_existing && start_row_boundary && end_row_boundary && (thread_meta_count == 0);

    bool BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    bool BMTB_nz_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id);
    bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);
    bool BMW_nz_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id);

    bool balanced_interval_and_max_padding_flag = false;
    for (int i = 0; i < former_operator.size(); i++)
    {
        if (this->former_operator[i]->get_name().find("balanced_interval") != string::npos)
        {
            balanced_interval_and_max_padding_flag = true;
        }
    }
    balanced_interval_and_max_padding_flag = balanced_interval_and_max_padding_flag & this->is_col_padding_with_row_max_size_with_empty_row;
    // row_indices 和 nz_indices 只能同时存在或不存在
    assert(BMTB_row_begin_existing ^ BMTB_nz_begin_existing == 0);
    assert(BMW_row_begin_existing ^ BMW_nz_begin_existing == 0);

    // 如果是relative的，那么必须有线程块级别的切分
    if (this->row_index_is_relative_to_parent == true)
    {
        necessary_data = necessary_data & (BMTB_row_begin_existing | BMW_row_begin_existing);
    }

    if (this->nz_index_is_relative_to_parent == true)
    {
        necessary_data = necessary_data & (BMTB_nz_begin_existing | BMW_nz_begin_existing);
    }

    // 如果是padding，之前必须没有BMTB、BMW的切分
    if (this->is_row_padding == true)
    {
        if (BMTB_row_begin_existing == true || BMW_row_begin_existing == true)
        {
            necessary_data = false;
        }
    }

    // 检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    // 如果之前存在BMTB、BMW的切分，那么必须是行切分
    bool valid_BMTB_blocking = false;
    if (BMTB_row_begin_existing == true)
    {
        valid_BMTB_blocking = has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id);
        necessary_data = necessary_data & valid_BMTB_blocking;
    }

    bool valid_BMW_blocking = false;
    if (BMW_row_begin_existing == true)
    {
        valid_BMW_blocking = has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, WARP_META, this->target_matrix_id);
        necessary_data = necessary_data & valid_BMW_blocking;
    }

    if (necessary_data == true && this->is_row_padding == true)
    {
        necessary_data = necessary_data & padding_rate_valid_row_direction(this->meta_data_set_ptr, this->fixed_row_block_size, this->target_matrix_id);
    }

    // 当前内容全部存在
    if (necessary_data == true && interlance_storage_existing == false && balanced_interval_and_max_padding_flag == false)
    {
        return true;
    }

    return false;
}

// 执行
void fixed_interval_row_direction_thread_blocking_operator::run(bool check)
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
    bool BMW_row_begin_existing = this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id);

    // 首先查看当前是否有BMTB、BMW级别的内容
    if (BMW_row_begin_existing == true)
    {
        if (check)
        {
            shared_ptr<universal_array> warp_first_row_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> warp_first_nz_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

            assert(warp_first_row_ptr->get_len() == warp_first_nz_ptr->get_len());
        }
        if (is_col_padding_with_row_max_size_with_empty_row == true)
        {
            shared_ptr<modify_col_indices_by_col_pad_parent_blk_to_max_row_size> col_padding_transform_ptr(new modify_col_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, WARP_META, this->target_matrix_id, true));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_col_pad_parent_blk_to_max_row_size> val_padding_transform_ptr(new modify_vals_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, WARP_META, this->target_matrix_id, true));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            shared_ptr<modify_row_indices_by_col_pad_parent_blk_to_max_row_size> row_padding_transform_ptr(new modify_row_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, WARP_META, this->target_matrix_id, true));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));

            vector<string> item_name_warp = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
            // 先删除之前已有的item
            for (int i = 0; i < item_name_warp.size(); i++)
            {
                shared_ptr<remove_item_of_metadata> remove_warp_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_warp[i], WARP_META));
                remove_warp_item_ptr->run(check);
                this->set_transform_seq(get_record_item_of_a_transform_step(remove_warp_item_ptr));
            }
            if (BMTB_row_begin_existing == true)
            {
                vector<string> item_name_tblock = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
                for (int i = 0; i < item_name_tblock.size(); i++)
                {
                    shared_ptr<remove_item_of_metadata> remove_tblock_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_tblock[i], TBLOCK_META));
                    remove_tblock_item_ptr->run(check);
                    this->set_transform_seq(get_record_item_of_a_transform_step(remove_tblock_item_ptr));
                }
            }
            // 重执行
            for (int i = 0; i < former_operator.size(); i++)
            {
                shared_ptr<basic_operator> rerun_operator = former_operator[i];
                int prev_size = rerun_operator->get_data_transform_sequence().size();
                rerun_operator->set_padding_to_false();
                rerun_operator->run(check);
                // 重执行的记录更新，只记录新执行的
                for (int j = prev_size; j < rerun_operator->get_data_transform_sequence().size(); j++)
                {
                    this->set_transform_seq(rerun_operator->get_data_transform_sequence()[j]);
                }
            }
        }

        if (this->is_col_padding_with_col_size == true)
        {
            shared_ptr<modify_col_indices_by_col_pad_in_sub_matrix> col_padding_transform_ptr(new modify_col_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_col_pad_in_sub_matrix> val_padding_transform_ptr(new modify_vals_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            shared_ptr<modify_row_indices_by_col_pad_in_sub_matrix> row_padding_transform_ptr(new modify_row_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));

            vector<string> item_name_warp = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
            // 先删除之前已有的item
            for (int i = 0; i < item_name_warp.size(); i++)
            {
                shared_ptr<remove_item_of_metadata> remove_warp_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_warp[i], WARP_META));
                remove_warp_item_ptr->run(check);
                this->set_transform_seq(get_record_item_of_a_transform_step(remove_warp_item_ptr));
            }
            if (BMTB_row_begin_existing == true)
            {
                vector<string> item_name_tblock = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
                for (int i = 0; i < item_name_tblock.size(); i++)
                {
                    shared_ptr<remove_item_of_metadata> remove_tblock_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_tblock[i], TBLOCK_META));
                    remove_tblock_item_ptr->run(check);
                    this->set_transform_seq(get_record_item_of_a_transform_step(remove_tblock_item_ptr));
                }
            }
            // 重执行
            for (int i = 0; i < former_operator.size(); i++)
            {
                shared_ptr<basic_operator> rerun_operator = former_operator[i];
                int prev_size = rerun_operator->get_data_transform_sequence().size();
                rerun_operator->set_padding_to_false();
                rerun_operator->run(check);
                // 重执行的记录更新，只记录新执行的
                for (int j = prev_size; j < rerun_operator->get_data_transform_sequence().size(); j++)
                {
                    this->set_transform_seq(rerun_operator->get_data_transform_sequence()[j]);
                }
            }
        }

        // 获得thread级别的首行绝对索引
        shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMW> get_absolute_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMT_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_row_ptr));

        // 首行相对索引
        if (this->row_index_is_relative_to_parent == true)
        {
            // 获得相对行索引
            shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW> get_relative_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            get_relative_BMT_start_row_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_row_ptr));
        }

        // thread级别的绝对非零元索引
        shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW> get_absolute_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMT_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_nz_ptr));

        // 相对非零元索引
        if (this->nz_index_is_relative_to_parent == true)
        {
            // 获得相对非零元索引
            shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW> get_relative_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            get_relative_BMT_start_nz_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_nz_ptr));
        }

        // 每一个BMW中BMT的相对索引
        shared_ptr<get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction> get_begin_BMT_of_BMW_ptr(new get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction(this->meta_data_set_ptr, WARP_META, this->target_matrix_id));
        get_begin_BMT_of_BMW_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMT_of_BMW_ptr));

        if (this->is_col_padding_with_row_max_size_with_empty_row == true && this->fixed_row_block_size == 1)
        {
            shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_GLOBAL_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, WARP_META, this->target_matrix_id, true));
            get_BMT_size_of_GLOBAL_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_GLOBAL_ptr));
        }
    }
    else if (BMTB_row_begin_existing == true)
    {
        if (check)
        {
            shared_ptr<universal_array> tblock_first_row_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> tblock_first_nz_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

            assert(tblock_first_row_ptr->get_len() == tblock_first_nz_ptr->get_len());
        }

        if (this->is_col_padding_with_row_max_size_with_empty_row == true)
        {
            shared_ptr<modify_col_indices_by_col_pad_parent_blk_to_max_row_size> col_padding_transform_ptr(new modify_col_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id, true));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_col_pad_parent_blk_to_max_row_size> val_padding_transform_ptr(new modify_vals_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id, true));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            shared_ptr<modify_row_indices_by_col_pad_parent_blk_to_max_row_size> row_padding_transform_ptr(new modify_row_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id, true));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));

            vector<string> item_name_tblock = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
            // 先删除之前已有的item
            for (int i = 0; i < item_name_tblock.size(); i++)
            {
                shared_ptr<remove_item_of_metadata> remove_tblock_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_tblock[i], TBLOCK_META));
                remove_tblock_item_ptr->run(check);
                this->set_transform_seq(get_record_item_of_a_transform_step(remove_tblock_item_ptr));
            }
            for (int i = 0; i < former_operator.size(); i++)
            {
                shared_ptr<basic_operator> rerun_operator = former_operator[i];
                int prev_size = rerun_operator->get_data_transform_sequence().size();
                rerun_operator->set_padding_to_false();
                rerun_operator->run(check);
                // 重执行的记录更新，只记录新执行的
                for (int j = prev_size; j < rerun_operator->get_data_transform_sequence().size(); j++)
                {
                    this->set_transform_seq(rerun_operator->get_data_transform_sequence()[j]);
                }
            }
        }

        if (this->is_col_padding_with_col_size == true)
        {

            shared_ptr<modify_col_indices_by_col_pad_in_sub_matrix> col_padding_transform_ptr(new modify_col_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_col_pad_in_sub_matrix> val_padding_transform_ptr(new modify_vals_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            shared_ptr<modify_row_indices_by_col_pad_in_sub_matrix> row_padding_transform_ptr(new modify_row_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));

            vector<string> item_name_tblock = this->meta_data_set_ptr->all_item_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
            // 先删除之前已有的item
            for (int i = 0; i < item_name_tblock.size(); i++)
            {
                shared_ptr<remove_item_of_metadata> remove_tblock_item_ptr(new remove_item_of_metadata(this->meta_data_set_ptr, this->target_matrix_id, item_name_tblock[i], TBLOCK_META));
                remove_tblock_item_ptr->run(check);
                this->set_transform_seq(get_record_item_of_a_transform_step(remove_tblock_item_ptr));
            }
            for (int i = 0; i < former_operator.size(); i++)
            {
                shared_ptr<basic_operator> rerun_operator = former_operator[i];
                int prev_size = rerun_operator->get_data_transform_sequence().size();
                rerun_operator->set_padding_to_false();
                rerun_operator->run(check);
                // 重执行的记录更新，只记录新执行的
                for (int j = prev_size; j < rerun_operator->get_data_transform_sequence().size(); j++)
                {
                    this->set_transform_seq(rerun_operator->get_data_transform_sequence()[j]);
                }
            }
        }

        // 获得thread级别的首行绝对索引
        shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB> get_absolute_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMT_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_row_ptr));

        // 首行相对索引
        if (this->row_index_is_relative_to_parent == true)
        {
            // 获得相对行索引
            shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB> get_relative_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            get_relative_BMT_start_row_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_row_ptr));
        }

        // thread级别的绝对非零元索引
        shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB> get_absolute_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMT_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_nz_ptr));

        // 相对非零元索引
        if (this->nz_index_is_relative_to_parent == true)
        {
            // 获得相对非零元索引
            shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB> get_relative_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            get_relative_BMT_start_nz_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMT_start_nz_ptr));
        }

        // 每一个BMTB中BMW的相对索引
        shared_ptr<get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction> get_begin_BMT_of_BMTB_ptr(new get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id));
        get_begin_BMT_of_BMTB_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMT_of_BMTB_ptr));

        if (this->is_col_padding_with_row_max_size_with_empty_row == true && this->fixed_row_block_size == 1)
        {
            shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_GLOBAL_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id, true));
            get_BMT_size_of_GLOBAL_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_GLOBAL_ptr));
        }
    }
    else
    {
        if (check)
        {
            assert(this->row_index_is_relative_to_parent == false);
            assert(this->nz_index_is_relative_to_parent == false);
        }

        // 可能会需要行方向的padding
        if (this->is_row_padding == true)
        {
            shared_ptr<modify_col_indices_by_row_pad_in_sub_matrix> col_padding_transform_ptr(new modify_col_indices_by_row_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_row_pad_in_sub_matrix> val_padding_transform_ptr(new modify_vals_by_row_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            // 执行行padding
            shared_ptr<modify_row_indices_by_row_pad_in_sub_matrix> row_padding_transform_ptr(new modify_row_indices_by_row_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));
        }

        if (this->is_col_padding_with_row_max_size_with_empty_row == true)
        {
            shared_ptr<modify_col_indices_by_col_pad_parent_blk_to_max_row_size> col_padding_transform_ptr(new modify_col_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, GLOBAL_META, this->target_matrix_id, true));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_col_pad_parent_blk_to_max_row_size> val_padding_transform_ptr(new modify_vals_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, GLOBAL_META, this->target_matrix_id, true));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            shared_ptr<modify_row_indices_by_col_pad_parent_blk_to_max_row_size> row_padding_transform_ptr(new modify_row_indices_by_col_pad_parent_blk_to_max_row_size(this->meta_data_set_ptr, GLOBAL_META, this->target_matrix_id, true));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));
        }

        if (this->is_col_padding_with_col_size)
        {
            shared_ptr<modify_col_indices_by_col_pad_in_sub_matrix> col_padding_transform_ptr(new modify_col_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            col_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(col_padding_transform_ptr));

            shared_ptr<modify_vals_by_col_pad_in_sub_matrix> val_padding_transform_ptr(new modify_vals_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            val_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(val_padding_transform_ptr));

            shared_ptr<modify_row_indices_by_col_pad_in_sub_matrix> row_padding_transform_ptr(new modify_row_indices_by_col_pad_in_sub_matrix(this->meta_data_set_ptr, this->target_matrix_id, this->col_size));
            row_padding_transform_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(row_padding_transform_ptr));
        }

        // 两个，row和nz绝对索引，并且这里不可能出现相对索引
        shared_ptr<get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction> get_absolute_BMT_start_row_ptr(new get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMT_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_row_ptr));

        shared_ptr<get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction> get_absolute_BMT_start_nz_ptr(new get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMT_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMT_start_nz_ptr));
        if (this->is_col_padding_with_row_max_size_with_empty_row == true && this->fixed_row_block_size == 1)
        {
            shared_ptr<get_BMT_size_of_each_parent> get_BMT_size_of_GLOBAL_ptr(new get_BMT_size_of_each_parent(this->meta_data_set_ptr, GLOBAL_META, this->target_matrix_id, true));
            get_BMT_size_of_GLOBAL_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_BMT_size_of_GLOBAL_ptr));
        }

        // 如果padding过，行索引应该是间隔相同的，做一个检查
        if (check)
        {
            if (this->is_row_padding == true)
            {
                // 读出来行偏移
                shared_ptr<universal_array> BMT_begin_row_index_ptr = this->meta_data_set_ptr->get_element(THREAD_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
                // 查看最后两个行边界之间的距离
                unsigned long BMT_begin_row_index_len = BMT_begin_row_index_ptr->get_len();
                assert(BMT_begin_row_index_len >= 2);
                assert((BMT_begin_row_index_ptr->read_integer_from_arr(BMT_begin_row_index_len - 1) - BMT_begin_row_index_ptr->read_integer_from_arr(BMT_begin_row_index_len - 2)) == this->fixed_row_block_size);
            }
        }
    }
    if (check)
    {
        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices", this->target_matrix_id));
    }

    this->code_generator_ptr->open_spec_level_of_paral(THREAD_META);
    this->code_generator_ptr->set_thread_for_row(true);
    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> fixed_interval_row_direction_thread_blocking_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->fixed_row_block_size > 0);

    if (is_row_padding == true)
    {
        assert(row_index_is_relative_to_parent == false);
        assert(nz_index_is_relative_to_parent == false);
    }

    // 检查内容中不能空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回对应序列
    return this->transform_seq;
}
string fixed_interval_row_direction_thread_blocking_operator::convert_to_string()
{
    assert(this->fixed_row_block_size > 0);
    assert(this->target_matrix_id >= 0);

    if (is_row_padding == true)
    {
        assert(row_index_is_relative_to_parent == false);
        assert(nz_index_is_relative_to_parent == false);
    }

    string return_str = "fixed_interval_row_direction_thread_blocking_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + ",is_row_padding:" + to_string(this->is_row_padding) + "}";

    return return_str;
}
