#include "../operator.hpp"

fixed_interval_row_direction_tblock_blocking_operator::fixed_interval_row_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool is_padding)
:basic_operator("fixed_interval_row_direction_tblock_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);
    this->fixed_row_block_size = fixed_row_block_size;
    this->is_padding = is_padding;
}

fixed_interval_row_direction_tblock_blocking_operator::fixed_interval_row_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool is_padding, shared_ptr<operator_context> operator_history)
:basic_operator("fixed_interval_row_direction_tblock_blocking_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(fixed_row_block_size > 0);
    this->fixed_row_block_size = fixed_row_block_size;
    this->is_padding = is_padding;
}

fixed_interval_row_direction_tblock_blocking_operator::fixed_interval_row_direction_tblock_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_block_size, bool is_padding, shared_ptr<operator_context> operator_history)
:basic_operator("fixed_interval_row_direction_tblock_blocking_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this)fixed_interval_row_direction_tblock_blocking_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), fixed_row_block_size, is_padding, operator_history);
    this->code_generator_ptr = code_generator_ptr;
}



bool fixed_interval_row_direction_tblock_blocking_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
        vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);

    if(former_operator_implementing.size() == 0)
    {
        if(former_operator_distributing.size() == 0)
        {
            return true;
        }
    }
    // bool tblock_flag = false;
    // bool warp_flag = false;
    // bool thread_flag = false;
    // bool interlance_flag = false;
    // for (int i; i < former_operator_distributing.size(); i++)
    // {
    //     if (former_operator_distributing[i]->get_name().find("tblock") != string::npos)
    //     {
    //         tblock_flag = true;
    //     }
    //     if (former_operator_distributing[i]->get_name().find("warp") != string::npos)
    //     {
    //         warp_flag = true;
    //     }
    //     if (former_operator_distributing[i]->get_name().find("thread") != string::npos)
    //     {
    //         thread_flag = true;
    //     }
    //     if (former_operator_distributing[i]->get_name().find("interlance") != string::npos)
    //     {
    //         interlance_flag = true;
    //     }
    // }
    // if (tblock_flag == false && warp_flag == false && thread_flag == false && interlance_flag == false)
    // {
    //     return true;
    // }

    return false;
}



bool fixed_interval_row_direction_tblock_blocking_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->fixed_row_block_size > 0);
    assert(this->target_matrix_id >= 0);
    
    // 存在行、列、值三个数组
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    // meta_data_set中不存在TBLOCK_META,WARP_META,THREAD_META数据
    int tblock_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(TBLOCK_META, this->target_matrix_id);
    int warp_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(WARP_META, this->target_matrix_id);
    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);

    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if(this->meta_data_set_ptr->is_exist(GLOBAL_META,"nz_row_indices_after_interlance_storage",this->target_matrix_id)==true || 
    this->meta_data_set_ptr->is_exist(GLOBAL_META,"nz_col_indices_after_interlance_storage",this->target_matrix_id)==true || 
    this->meta_data_set_ptr->is_exist(GLOBAL_META,"nz_vals_after_interlance_storage",this->target_matrix_id)==true)
    {
        interlance_storage_existing = true;
    }

    // 三个数组都存在
    if (row_indices_existing == true && col_indices_existing == true && vals_existing == true && interlance_storage_existing == false)
    {
        // 不存在任何一种类型的
        if (tblock_meta_count == 0 && warp_meta_count == 0 && thread_meta_count == 0)
        {
            if (start_row_boundary == true && end_row_boundary == true)
            {
                if(this->is_padding == true)
                {
                    return padding_rate_valid_row_direction(this->meta_data_set_ptr, this->fixed_row_block_size, this->target_matrix_id);
                }
                else
                {
                    return true;
                }
            }
        }
    }
    
    return false;
}

// 执行
void fixed_interval_row_direction_tblock_blocking_operator::run(bool check)
{

    if(check)
    {
        assert(this->is_valid_according_to_metadata());
    }

    // 首先做一系列运行前检查，行列值数组长度相等
    shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();
    assert(row_index_ptr->get_len() == col_index_ptr->get_len());
    assert(col_index_ptr->get_len() == val_ptr->get_len());

    // 首先根据内容查看是否padding
    if (this->is_padding == true)
    {
        // 执行padding
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

    // 然后执行分块，首先产生行索引分块
    shared_ptr<get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction> get_begin_rows_tranform_ptr(new get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
    get_begin_rows_tranform_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_rows_tranform_ptr));

    // 如果padding过，行索引应该是间隔相同的，做一个检查
    if (this->is_padding == true)
    {
        // 读出来行偏移
        shared_ptr<universal_array> BMTB_begin_row_index_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
        // 查看最后两个行边界之间的距离
        unsigned long BMTB_begin_row_index_len = BMTB_begin_row_index_ptr->get_len();
        assert(BMTB_begin_row_index_len >= 2);
        assert((BMTB_begin_row_index_ptr->read_integer_from_arr(BMTB_begin_row_index_len - 1) - BMTB_begin_row_index_ptr->read_integer_from_arr(BMTB_begin_row_index_len - 2)) == this->fixed_row_block_size);
    }

    // 值索引的偏移
    shared_ptr<get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction> get_begin_nzs_tranform_ptr(new get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction(this->meta_data_set_ptr, this->target_matrix_id, this->fixed_row_block_size));
    get_begin_nzs_tranform_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_nzs_tranform_ptr));

    row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    col_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
    val_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> BMTB_begin_nz_index_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

    assert(row_index_ptr->get_len() == col_index_ptr->get_len());
    assert(col_index_ptr->get_len() == val_ptr->get_len());
    assert(row_index_ptr->get_len() == BMTB_begin_nz_index_ptr->read_integer_from_arr(BMTB_begin_nz_index_ptr->get_len() - 1));

    this->code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);
    this->is_run = true;
}


vector<shared_ptr<transform_step_record_item>> fixed_interval_row_direction_tblock_blocking_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->fixed_row_block_size > 0);
    
    // 检查内容中不能空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回对应序列
    return this->transform_seq;
}

string fixed_interval_row_direction_tblock_blocking_operator::convert_to_string()
{
    assert(this->fixed_row_block_size > 0);
    assert(this->target_matrix_id >= 0);

    string return_str = "fixed_interval_row_direction_tblock_blocking_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",fixed_row_block_size:" + to_string(this->fixed_row_block_size) + "}";

    return return_str;
}

