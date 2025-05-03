#include "../data_transform_step.hpp"

get_row_order_by_col::get_row_order_by_col(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("get_row_order_by_col", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    // 没有数据成员需要处理
    // 仅仅做一些检查
    this->target_matrix_id = target_matrix_id;
}

void get_row_order_by_col::run(bool check)
{
    if (check)
    {
        assert(target_matrix_id >= 0);
        assert(this->meta_data_set_ptr->check());
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        assert(!(this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id)));
    }

    shared_ptr<universal_array> row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> col_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();


    unsigned long min_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long max_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    
    unsigned long min_col_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_col_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long max_col_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_col_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);


    vector<unsigned long> row_index_order_by_col_vec = get_row_order_vec(row_index, col_index, min_col_index, max_col_index);

    // 首先创造一个通用数组
    shared_ptr<universal_array> original_row_index_ptr(new universal_array((void *)(&row_index_order_by_col_vec[0]), row_index_order_by_col_vec.size(), UNSIGNED_LONG));

    // 将数组中的内容放到metadata set中，先产生一个条目
    shared_ptr<meta_data_item> item_ptr(new meta_data_item(original_row_index_ptr, GLOBAL_META, "original_nz_row_indices", this->target_matrix_id));

    // 加入到元数据集中
    this->meta_data_set_ptr->add_element(item_ptr);
    this->is_run = true;
}

vector<shared_ptr<data_item_record>> get_row_order_by_col::get_source_data_item_ptr_in_data_transform_step()
{
    // input1:GLOBAL_META, "nz_row_indices", this->target_matrix_id
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true); 这里的执行的输入和输出可以在静态的方式中进行
    vector<shared_ptr<data_item_record>> return_vec;
    shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));

    return_vec.push_back(nz_row_indices_record);

    return return_vec;
}

vector<shared_ptr<data_item_record>> get_row_order_by_col::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // output2:GLOBAL_META, "original_nz_row_indices", this->target_matrix_id
    assert(this->target_matrix_id >= 0);
    // assert(this->is_run == true); 这里的执行的输入和输出可以在静态的方式中进行

    vector<shared_ptr<data_item_record>> return_vec;

    shared_ptr<data_item_record> original_nz_row_indices_record(new data_item_record(GLOBAL_META, "original_nz_row_indices", this->target_matrix_id));

    return_vec.push_back(original_nz_row_indices_record);

    return return_vec;
}

string get_row_order_by_col::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    // 打印名字和参数
    string return_str = "get_row_order_by_col::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}