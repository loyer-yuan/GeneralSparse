#include "../data_transform_step.hpp"

get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB::get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW)
    : basic_data_transform_step("get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(nnz_per_BMW > 0);

    this->target_matrix_id = target_matrix_id;
    this->nnz_per_BMW = nnz_per_BMW;
}

void get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->nnz_per_BMW > 0);
        // 检查
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_row_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", this->target_matrix_id));

    }
    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    // 取出所有的BMTB行偏移量
    shared_ptr<universal_array> BMTB_first_row_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> BMTB_first_nz_indices_ptr = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", this->target_matrix_id)->get_metadata_arr();

    // 计入输入数据
    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    shared_ptr<data_item_record> BMTB_first_row_indices_record(new data_item_record(TBLOCK_META, "first_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(BMTB_first_row_indices_record);

    shared_ptr<data_item_record> BMTB_first_nz_indices_record(new data_item_record(TBLOCK_META, "first_nz_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(BMTB_first_nz_indices_record);

 
    vector<unsigned long> BMW_begin_row_vec_relative_to_BMTB;

    unsigned int BMTB_id = 0;

    
    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i += this->nnz_per_BMW)
    {
        unsigned long next_BMTB_first_nz = BMTB_first_nz_indices_ptr->read_integer_from_arr(BMTB_id + 1);
        if (i >= next_BMTB_first_nz)
        {
            // 为了保证父块和子块切分间隔是整倍数关系
            assert(i == next_BMTB_first_nz);
            BMTB_id += 1;
        }

        unsigned long current_BMTB_first_row = BMTB_first_row_indices_ptr->read_integer_from_arr(BMTB_id);

        unsigned long row_index = nz_row_indices_ptr->read_integer_from_arr(i);
        BMW_begin_row_vec_relative_to_BMTB.push_back(row_index - current_BMTB_first_row);
    }

    if (check)
    {
        assert(BMW_begin_row_vec_relative_to_BMTB.size() > 0);
    }

    // 将相对索引放到metadata set中
    shared_ptr<universal_array> BMW_first_row_index_relative_to_BMTB_ptr(new universal_array(&(BMW_begin_row_vec_relative_to_BMTB[0]), BMW_begin_row_vec_relative_to_BMTB.size(), UNSIGNED_LONG));
    // 行偏移的相对索引
    shared_ptr<meta_data_item> BMW_first_row_index_relative_to_BMTB_item(new meta_data_item(BMW_first_row_index_relative_to_BMTB_ptr, WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
    // 将新的内容写到metadata set中
    this->meta_data_set_ptr->add_element(BMW_first_row_index_relative_to_BMTB_item);

    // 执行记录
    shared_ptr<data_item_record> BMW_first_row_index_relative_to_BMTB_record(new data_item_record(WARP_META, "first_row_indices_relative_to_BMTB", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMW_first_row_index_relative_to_BMTB_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB::get_source_data_item_ptr_in_data_transform_step()
{
    // 内容满足要求
    assert(target_matrix_id >= 0);
    assert(nnz_per_BMW > 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // 内容满足要求
    assert(target_matrix_id >= 0);
    assert(nnz_per_BMW > 0);
    assert(this->is_run == true);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB::convert_to_string()
{
    assert(target_matrix_id >= 0);
    assert(nnz_per_BMW > 0);

    string return_str = "get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",nnz_per_BMW:" + to_string(this->nnz_per_BMW) + "}";

    return return_str;
}