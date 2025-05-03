#include "../IO_of_reduction.hpp"

one_register_result_IO_of_reduction::one_register_result_IO_of_reduction(POS_TYPE pos, int count)
:basic_IO_of_reduction("one_register_result_IO_of_reduction", pos)
{
    // 创建一个变量，变量名是为one_register_result
    shared_ptr<var_name_token> IO_var_token(new var_name_token(convert_pos_type_to_string(pos) +"_" + "one_register_result", REGISTER_VAR_TYPE));
    assert(IO_var_token->static_check() == true);

    this->unbounded_var_name_vec.push_back(IO_var_token);
    this->count = count;
}



shared_ptr<var_name_token> one_register_result_IO_of_reduction::var_name_token_of_IO_register()
{
    assert(this->unbounded_var_name_vec.size() == 1);

    return this->unbounded_var_name_vec[0];
}

unsigned int one_register_result_IO_of_reduction::get_count()
{
    return this->count;
}