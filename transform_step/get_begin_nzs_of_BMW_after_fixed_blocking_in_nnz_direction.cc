#include "../data_transform_step.hpp"

get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction::get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW)
    : basic_data_transform_step("get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(this->meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(nnz_per_BMW > 0);

    this->target_matrix_id = target_matrix_id;
    this->nnz_per_BMW = nnz_per_BMW;
}

void get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->nnz_per_BMW > 0);
        // 检查
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());

        // 不能出现交错存储
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices_after_interlance_storage", this->target_matrix_id) == false);

        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    shared_ptr<universal_array> nz_row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

    // 用一个数组存储每一个BMW的起始非零元偏移量
    vector<unsigned long> BMW_begin_nz_vec;

    for (unsigned long i = 0; i < nz_row_indices_ptr->get_len(); i+= this->nnz_per_BMW)
    {
        BMW_begin_nz_vec.push_back(i);
    }
    BMW_begin_nz_vec.push_back(nz_row_indices_ptr->get_len());

    // 将新的内容放到metadata set中
    shared_ptr<universal_array> BMW_begin_nz_ptr(new universal_array(&(BMW_begin_nz_vec[0]), BMW_begin_nz_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> BMW_begin_nz_item(new meta_data_item(BMW_begin_nz_ptr, WARP_META, "first_nz_indices", this->target_matrix_id));
    this->meta_data_set_ptr->add_element(BMW_begin_nz_item);

    // 加入新的record
    shared_ptr<data_item_record> BMW_begin_nz_record(new data_item_record(WARP_META, "first_nz_indices", this->target_matrix_id));
    this->dest_data_item_ptr_vec.push_back(BMW_begin_nz_record);

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->nnz_per_BMW > 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->target_matrix_id >= 0);
    assert(this->nnz_per_BMW > 0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->nnz_per_BMW > 0);

    string return_str = "get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",nnz_per_BMW:" + to_string(this->nnz_per_BMW) + "}";

    return return_str;
}