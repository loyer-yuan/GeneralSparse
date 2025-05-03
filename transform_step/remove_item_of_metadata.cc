#include "../data_transform_step.hpp"

remove_item_of_metadata::remove_item_of_metadata(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, string item_name, POS_TYPE pos)
:basic_data_transform_step("remove_item_of_metadata", meta_data_set_ptr)
{

    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);
    assert(item_name.size()>0);
    assert(check_pos_type(pos) == true);
    this->target_matrix_id = target_matrix_id;
    this->item_name = item_name;
    this->pos = pos;
}

void remove_item_of_metadata::run(bool check)
{
    if (check)
    {
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        assert(this->target_matrix_id >= 0);
        assert(this->item_name.size() > 0);
        assert(this->meta_data_set_ptr->is_exist(pos, item_name, this->target_matrix_id));
    }

    this->meta_data_set_ptr->remove_element(pos, item_name, this->target_matrix_id);
    shared_ptr<data_item_record> remove_item_record(new data_item_record(pos, item_name, this->target_matrix_id));
    this->source_data_item_ptr_vec.push_back(remove_item_record);
        
    
    this->is_run = true;
}
vector<shared_ptr<data_item_record>> remove_item_of_metadata::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->target_matrix_id >= 0);
    assert(this->item_name.size()>0);
    assert(this->is_run == true);

    // 检查空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}
vector<shared_ptr<data_item_record>> remove_item_of_metadata::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->is_run == true);
    assert(this->target_matrix_id >= 0);
    return this->dest_data_item_ptr_vec;
}

string remove_item_of_metadata::convert_to_string()
{
    assert(this->target_matrix_id >= 0);
    assert(this->item_name.size()>0);

    string return_str = "remove_item_of_metadata::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ", pos:" + convert_pos_type_to_string(pos) + "}";

    return return_str;
}
