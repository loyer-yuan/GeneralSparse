#include "../IO_of_reduction.hpp"

two_register_result_IO_of_reduction::two_register_result_IO_of_reduction(POS_TYPE pos, int count)
:basic_IO_of_reduction("two_register_result_IO_of_reduction", pos)
{
    // 创建一个变量，变量名是为one_register_result
    shared_ptr<var_name_token> IO_var_token_1(new var_name_token("head_result", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> IO_var_token_2(new var_name_token("tail_result", REGISTER_VAR_TYPE));


    this->unbounded_var_name_vec.push_back(IO_var_token_1);
    this->unbounded_var_name_vec.push_back(IO_var_token_2);

    this->count = count;
}



vector<shared_ptr<var_name_token>> two_register_result_IO_of_reduction::var_names_token_of_IO_register()
{
    assert(this->unbounded_var_name_vec.size() == 2);

    return this->unbounded_var_name_vec;
}

unsigned int two_register_result_IO_of_reduction::get_count()
{
    return this->count;
}