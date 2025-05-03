#include "../operator.hpp"

merge_path_thread_operator::merge_path_thread_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size)
:basic_operator("merge_path_thread_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(work_size > 0);
    this->work_size = work_size;
}

merge_path_thread_operator::merge_path_thread_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size, shared_ptr<operator_context> operator_history)
:basic_operator("merge_path_thread_operator", meta_data_set_ptr, DISTRIBUTING_OP, target_matrix_id)
{
    assert(target_matrix_id >= 0);
    assert(work_size > 0);
    this->work_size = work_size;
}

merge_path_thread_operator::merge_path_thread_operator(shared_ptr<code_generator> code_generator_ptr, int work_size, shared_ptr<operator_context> operator_history)
:basic_operator("merge_path_thread_operator", code_generator_ptr->get_metadata_set(), DISTRIBUTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    new(this)merge_path_thread_operator(code_generator_ptr->get_metadata_set(), code_generator_ptr->get_sub_matrix_id(), work_size, operator_history);
    this->code_generator_ptr = code_generator_ptr;
}



bool merge_path_thread_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    vector<shared_ptr<basic_operator>> former_operator_distributing = operator_history->read_operator_context_arr(DISTRIBUTING_OP, this->target_matrix_id);
    vector<shared_ptr<basic_operator>> former_operator_implementing = operator_history->read_operator_context_arr(IMPLEMENTING_OP, this->target_matrix_id);
    bool thread_flag = false;
    bool col_direction_flag = false;
    bool interlance_flag = false;
    if(former_operator_implementing.size() == 0)
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
        }
    }

    if(thread_flag == false && col_direction_flag == false && interlance_flag == false)
    {
        return true;
    }
   

    return false;
}



bool merge_path_thread_operator::is_valid_according_to_metadata()
{
    assert(this->meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(this->work_size > 0);
    assert(this->target_matrix_id >= 0);
    
    // 存在行、列、值三个数组
    bool row_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id);
    bool col_indices_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id);
    bool vals_existing = this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id);
    bool start_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id);
    bool end_row_boundary = this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id);

    int thread_meta_count = this->meta_data_set_ptr->count_of_metadata_of_diff_pos(THREAD_META, this->target_matrix_id);

    //检查是否存在交错存储
    bool interlance_storage_existing = false;
    if(this->meta_data_set_ptr->is_exist(GLOBAL_META,"nz_row_indices_after_interlance_storage",this->target_matrix_id)==true || 
    this->meta_data_set_ptr->is_exist(GLOBAL_META,"nz_col_indices_after_interlance_storage",this->target_matrix_id)==true || 
    this->meta_data_set_ptr->is_exist(GLOBAL_META,"nz_vals_after_interlance_storage",this->target_matrix_id)==true)
    {
        interlance_storage_existing = true;
    }

    if (row_indices_existing == true && col_indices_existing == true && vals_existing == true && interlance_storage_existing == false)
    {
        if (thread_meta_count == 0)
        {
            if (start_row_boundary == true && end_row_boundary == true)
            {
                return true;
            }
        }
    }
    
    return false;
}

// 执行
void merge_path_thread_operator::run(bool check)
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


    // 然后执行分块，首先产生行索引分块
    shared_ptr<get_begin_rows_of_level_after_merge_path> get_begin_rows_tranform_ptr(new get_begin_rows_of_level_after_merge_path(this->meta_data_set_ptr, this->target_matrix_id, THREAD_META, this->work_size));
    get_begin_rows_tranform_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_rows_tranform_ptr));


    // 值索引的偏移
    shared_ptr<get_begin_nzs_of_level_after_merge_path> get_begin_nzs_tranform_ptr(new get_begin_nzs_of_level_after_merge_path(this->meta_data_set_ptr, this->target_matrix_id, THREAD_META, this->work_size));
    get_begin_nzs_tranform_ptr->run(check);
    this->set_transform_seq(get_record_item_of_a_transform_step(get_begin_nzs_tranform_ptr));


    this->code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);
    this->is_run = true;
}


vector<shared_ptr<transform_step_record_item>> merge_path_thread_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);
    assert(this->is_run == true);
    assert(this->work_size > 0);
    
    // 检查内容中不能空指针
    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回对应序列
    return this->transform_seq;
}

string merge_path_thread_operator::convert_to_string()
{
    assert(this->work_size > 0);
    assert(this->target_matrix_id >= 0);

    string return_str = "merge_path_thread_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + ",work_size:" + to_string(this->work_size) + "}";

    return return_str;
}

