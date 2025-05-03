#include "../operator.hpp"

fixed_interval_row_direction_warp_blocking_operator::fixed_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding)
    : basic_operator("fixed_interval_row_direction_warp_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    // 如果padding了，那就一定没有BMTB的分块，也就一定没有相对索引的生成要求
    if (is_padding == true)
    {
        assert(row_index_is_relative_to_BMTB == false);
        assert(nz_index_is_relative_to_BMTB == false);
    }

    if (row_index_is_relative_to_BMTB == true || nz_index_is_relative_to_BMTB == true)
    {
        assert(is_padding == false);
    }

    this->fixed_row_block_size = fixed_row_block_size;
    this->row_index_is_relative_to_BMTB = row_index_is_relative_to_BMTB;
    this->nz_index_is_relative_to_BMTB = nz_index_is_relative_to_BMTB;
    this->is_padding = is_padding;
}

fixed_interval_row_direction_warp_blocking_operator::fixed_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_row_direction_warp_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);

    // 如果padding了，那就一定没有BMTB的分块，也就一定没有相对索引的生成要求
    if (is_padding == true)
    {
        assert(row_index_is_relative_to_BMTB == false);
        assert(nz_index_is_relative_to_BMTB == false);
    }

    if (row_index_is_relative_to_BMTB == true || nz_index_is_relative_to_BMTB == true)
    {
        assert(is_padding == false);
    }

    this->fixed_row_block_size = fixed_row_block_size;
    this->row_index_is_relative_to_BMTB = row_index_is_relative_to_BMTB;
    this->nz_index_is_relative_to_BMTB = nz_index_is_relative_to_BMTB;
    this->is_padding = is_padding;
}


fixed_interval_row_direction_warp_blocking_operator::fixed_interval_row_direction_warp_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding, shared_ptr<operator_context> operator_history)
    : basic_operator("fixed_interval_row_direction_warp_blocking_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this)fixed_interval_row_direction_warp_blocking_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), fixed_row_block_size, row_index_is_relative_to_BMTB, nz_index_is_relative_to_BMTB, is_padding, operator_history);
    this->code_generator_ptr = code_generator_ptr;
}


bool fixed_interval_row_direction_warp_blocking_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);

    bool col_direction_flag = false;
    bool thread_flag = false;
    bool warp_flag = false;
    bool interlance_flag = false;
    if (former_operator_implementing.size() == 0)
    {
        for (int i = 0; i < former_operator_distributing.size(); i++)
        {
            if (former_operator_distributing[i]->get_name().find("thread") != string::npos)
            {
                thread_flag = true;
            }
            if (former_operator_distributing[i]->get_name().find("warp") != string::npos)
            {
                warp_flag = true;
            }
            if (former_operator_distributing[i]->get_name().find("col") != string::npos)
            {
                col_direction_flag = true;
            }
            if (former_operator_distributing[i]->get_name().find("interlance") != string::npos)
            {
                interlance_flag = true;
            }
        }
        if (thread_flag == false && warp_flag == false && col_direction_flag == false && interlance_flag == false)
        {
            return true;
        }
    }

    return false;
}

bool fixed_interval_row_direction_warp_blocking_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->fixed_row_block_size > 0);
    assert(this->target_matrix_id >= 0);

    if (is_padding == true)
    {
        assert(row_index_is_relative_to_BMTB == false);
        assert(nz_index_is_relative_to_BMTB == false);
    }

    if (row_index_is_relative_to_BMTB == true || nz_index_is_relative_to_BMTB == true)
    {
        assert(is_padding == false);
    }

    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    int warp_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);
    int tblock_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);

    bool BMTB_row_begin_existing = true;
    bool BMTB_nz_begin_existing = true;

    // 如果是relative的，那么必须有线程块级别的切分
    if (this->row_index_is_relative_to_BMTB == true)
    {
        // BMTB粒度的两个数据必须存在
        BMTB_row_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id);
    }

    if (this->nz_index_is_relative_to_BMTB == true)
    {
        BMTB_nz_begin_existing = this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id);
    }

    bool valid_BMTB_blocking = true;

    // 如果之前存在BMTB的切分，那么必须是行切分
    if (tblock_meta_count != 0)
    {
        // 必须存在行偏移
        valid_BMTB_blocking = has_row_direction_blocking_in_specific_level(this->meta_data_set_ptr, TBLOCK_META, this->target_matrix_id);
    }

    // 如果是padding，之前必须没有BMTB的切分
    bool valid_padding = true;

    if (this->is_padding == true && tblock_meta_count != 0)
    {
        valid_padding = false;
    }

    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices_after_interlance_storage", this->target_matrix_id) == true ||
        this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals_after_interlance_storage", this->target_matrix_id) == true)
    {
        interlance_storage_existing = true;
    }

    // 当前内容全部存在
    bool needed_data = row_indices_existing && col_indices_existing && vals_existing && start_row_boundary && end_row_boundary && (warp_meta_count == 0) && (thread_meta_count == 0);
    needed_data = needed_data && BMTB_nz_begin_existing && BMTB_row_begin_existing && valid_BMTB_blocking && valid_padding && interlance_storage_existing == false;

    if (needed_data == true && this->is_padding == true)
    {
        needed_data = needed_data & padding_rate_valid_row_direction(this->meta_data_set_ptr, this->fixed_row_block_size, this->target_matrix_id);
    }

    if (needed_data == true)
    {
        return true;
    }

    return false;
}

void fixed_interval_row_direction_warp_blocking_operator::run(bool check)
{
    if (check)
    {
        assert(this->is_valid_according_to_metadata());

        if (is_padding == true)
        {
            assert(row_index_is_relative_to_BMTB == false);
            assert(nz_index_is_relative_to_BMTB == false);
        }

        if (row_index_is_relative_to_BMTB == true || nz_index_is_relative_to_BMTB == true)
        {
            assert(is_padding == false);
        }
    }

    // 首先做一系列运行前检查，行列值数组长度相等
    if (check)
    {
        shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();
        if (check)
        {
            assert(row_index_ptr->get_len() == col_index_ptr->get_len());
            assert(col_index_ptr->get_len() == val_ptr->get_len());
        }
    }

    // 首先查看当前是否有BMTB级别的内容
    if (this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id) == true)
    {
        if (check)
        {
            assert(this->is_padding == false);
            assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id) == true);

            shared_ptr<universal_array> tblock_first_row_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
            shared_ptr<universal_array> tblock_first_nz_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

            assert(tblock_first_row_ptr->get_len() == tblock_first_nz_ptr->get_len());
        }

        // 获得warp级别的首行绝对索引
        shared_ptr<get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB> get_absolute_BMW_start_row_ptr(new get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMW_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_row_ptr));

        // warp级别的首相相对索引
        if (this->row_index_is_relative_to_BMTB == true)
        {
            // 获得相对行索引
            shared_ptr<get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB> get_relative_BMW_start_row_ptr(new get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            get_relative_BMW_start_row_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMW_start_row_ptr));
        }

        // warp级别的绝对非零元索引
        shared_ptr<get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB> get_absolute_BMW_start_nz_ptr(new get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMW_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_nz_ptr));

        // warp级别的相对非零元索引
        if (this->nz_index_is_relative_to_BMTB == true)
        {
            // 获得相对非零元索引
            shared_ptr<get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB> get_relative_BMW_start_nz_ptr(new get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
            get_relative_BMW_start_nz_ptr->run(check);
            this->set_transform_seq(get_record_item_of_a_transform_step(get_relative_BMW_start_nz_ptr));
        }

        // 每一个BMTB中BMW的相对索引
        shared_ptr<get_begin_BMWs_of_BMTB_after_blocking_in_row_direction> get_begin_BMWs_of_BMTB_after_blocking_in_row_direction_ptr(new get_begin_BMWs_of_BMTB_after_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id));
        get_begin_BMWs_of_BMTB_after_blocking_in_row_direction_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_BMWs_of_BMTB_after_blocking_in_row_direction_ptr));
        if (check)
        {
            // 检查，BMTB的最后一个BMW的索引和BMW的数量是一致的
            assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_BMW_indices", this->target_matrix_id));
            shared_ptr<universal_array> first_BMW_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_BMW_indices", this->target_matrix_id)->get_metadata_arr();

            assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id));
            // 实际的长度比BMW的数量多一个，是CSR-like的索引
            shared_ptr<universal_array> warp_first_nz_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

            if (first_BMW_indices_ptr->read_integer_from_arr(first_BMW_indices_ptr->get_len() - 1) + 1 != warp_first_nz_indices_ptr->get_len())
            {
                cout << "fixed_interval_row_direction_warp_blocking_operator::run(): first_BMW_indices_ptr->read_integer_from_arr(first_BMW_indices_ptr->get_len() - 1)=" << first_BMW_indices_ptr->read_integer_from_arr(first_BMW_indices_ptr->get_len() - 1) << ", warp_first_nz_indices_ptr->get_len()=" << warp_first_nz_indices_ptr->get_len() << endl;
                assert(false);
            }
        }
    }
    else
    {
        if (check)
        {
            assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id) == false);
            assert(this->row_index_is_relative_to_BMTB == false);
            assert(this->nz_index_is_relative_to_BMTB == false);
        }

        // 可能会需要行方向的padding
        if (this->is_padding == true)
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

        // 两个，row和nz绝对索引，并且这里不可能出现相对索引
        // 获得warp级别的首行绝对索引
        shared_ptr<get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB> get_absolute_BMW_start_row_ptr(new get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMW_start_row_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_row_ptr));

        // warp首非零元索引
        shared_ptr<get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB> get_absolute_BMW_start_nz_ptr(new get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
        get_absolute_BMW_start_nz_ptr->run(check);
        this->set_transform_seq(get_record_item_of_a_transform_step(get_absolute_BMW_start_nz_ptr));

        if (check)
        {
            // 如果padding过，行索引应该是间隔相同的，做一个检查
            if (this->is_padding == true)
            {
                // 读出来行偏移
                shared_ptr<universal_array> BMW_begin_row_index_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
                // 查看最后两个行边界之间的距离
                unsigned long BMW_begin_row_index_len = BMW_begin_row_index_ptr->get_len();
                assert(BMW_begin_row_index_len >= 2);
                assert((BMW_begin_row_index_ptr->read_integer_from_arr(BMW_begin_row_index_len - 1) - BMW_begin_row_index_ptr->read_integer_from_arr(BMW_begin_row_index_len - 2)) == this->fixed_row_block_size);
            }
        }
    }
    if (check)
    {
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices", this->target_matrix_id));

        // 长度相同
        shared_ptr<universal_array> warp_first_nz_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();
        shared_ptr<universal_array> warp_first_row_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();

        assert(warp_first_nz_indices_ptr->get_len() == warp_first_row_indices_ptr->get_len());
        shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();
        assert(warp_first_nz_indices_ptr->read_integer_from_arr(warp_first_nz_indices_ptr->get_len() - 1) == val_ptr->get_len());

        // 查看相对索引
        if (this->row_index_is_relative_to_BMTB == true)
        {
            // 存在对应元素
            assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
            // 相对行索引
            shared_ptr<universal_array> warp_relative_first_row_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id)->get_metadata_arr();
            // 长度比绝对索引少一个
            assert(warp_relative_first_row_indices_ptr->get_len() == warp_first_row_indices_ptr->get_len() - 1);
        }

        if (this->nz_index_is_relative_to_BMTB == true)
        {
            assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices_relative_to_BMTB", this->target_matrix_id));
            // 相对非零元索引
            shared_ptr<universal_array> warp_relative_first_nz_indices_ptr = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices_relative_to_BMTB", this->target_matrix_id)->get_metadata_arr();
            assert(warp_relative_first_nz_indices_ptr->get_len() == warp_first_nz_indices_ptr->get_len() - 1);
        }
    }
    // 进行检查，绝对索引的长度是一样
    this->code_generator_ptr->open_spec_level_of_paral(WARP_META);
    this->is_run = true;
}

vector<shared_ptr<transform_step_record_item>> fixed_interval_row_direction_warp_blocking_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->fixed_row_block_size > 0);

    if (is_padding == true)
    {
        assert(row_index_is_relative_to_BMTB == false);
        assert(nz_index_is_relative_to_BMTB == false);
    }

    if (row_index_is_relative_to_BMTB == true || nz_index_is_relative_to_BMTB == true)
    {
        assert(is_padding == false);
    }

    // 没有空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    return this->transform_seq;
}

string fixed_interval_row_direction_warp_blocking_operator::convert_to_string()
{
    assert(this->fixed_row_block_size > 0);
    assert(this->target_matrix_id >= 0);

    if (is_padding == true)
    {
        assert(row_index_is_relative_to_BMTB == false);
        assert(nz_index_is_relative_to_BMTB == false);
    }

    if (row_index_is_relative_to_BMTB == true || nz_index_is_relative_to_BMTB == true)
    {
        assert(is_padding == false);
    }

    string return_str = "fixed_interval_row_direction_warp_blocking_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + ",is_padding:" + to_string(this->is_padding) + "}";

    return return_str;
}