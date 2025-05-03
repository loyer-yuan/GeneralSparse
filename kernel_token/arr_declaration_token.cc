#include "../kernel_generator.h"

arr_declaration_token::arr_declaration_token(shared_ptr<data_type_token> data_type_declare, shared_ptr<var_name_token> mem_ptr_name, shared_ptr<math_expr_token> mem_size)
    : basic_token(false, ARR_DECLARATION_TOKEN_TYPE)
{
    assert(mem_ptr_name != NULL);
    assert(mem_size != NULL);

    this->token_of_child_map["data_type"] = data_type_declare;
    this->token_of_child_map["mem_ptr_name"] = mem_ptr_name;
    this->token_of_child_map["mem_size"] = mem_size;
}

string arr_declaration_token::run()
{
    assert(this->child_is_exist("mem_ptr_name") == true);
    assert(this->child_is_exist("mem_size") == true);
    assert(this->static_check() == true);
    string return_code_str = this->token_of_child_map["data_type"]->run() + " " + this->token_of_child_map["mem_ptr_name"]->run() + "[" + this->token_of_child_map["mem_size"]->run() + "];\n";

    if (this->token_of_child_map["data_type"]->run() == "float" || this->token_of_child_map["data_type"]->run() == "half" || this->token_of_child_map["data_type"]->run() == "double")
    {
        return_code_str += "for(int i = 0; i < " + this->token_of_child_map["mem_size"]->run() + "; i++)\n";
        return_code_str += "{\n";
        return_code_str += this->token_of_child_map["mem_ptr_name"]->run() + "[i] = 0;\n";
        return_code_str += "}\n";
    }

    return return_code_str;
}

bool arr_declaration_token::static_check()
{
    assert(this->child_is_exist("mem_ptr_name") == true);
    assert(this->child_is_exist("mem_size") == true);

    // 三个内容的token类型
    if (!(this->token_of_child_map["mem_ptr_name"]->get_token_type() == VAR_NAME_TOKEN_TYPE && this->token_of_child_map["mem_size"]->get_token_type() == MATH_EXPR_TOKEN_TYPE))
    {
        cout << "arr_declaration_token::static_check(): invalid var_name of child" << endl;
        return false;
    }



    // // 检查第二个元素
    // shared_ptr<var_name_token> mem_ptr_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["mem_ptr_name"]);
    // assert(mem_ptr_name != NULL);

    // // 第二个变量是GLOBAL和SHARED类型，如果
    // if (!(mem_ptr_name->get_var_type() == SHARED_MEM_VAR_TYPE || mem_ptr_name->get_var_type() == GLOBAL_MEM_VAR_TYPE) || mem_ptr_name->static_check() == false)
    // {
    //     cout << "arr_declaration_token::static_check(): mem_ptr_name is not SHARED_MEM and GLOBAL_MEM" << endl;
    //     return false;
    // }

    // 将第三个元素取出
    shared_ptr<basic_token> basic_mem_index = this->token_of_child_map["mem_size"];
    assert(basic_mem_index != NULL);

    if (basic_mem_index->get_token_type() == MATH_EXPR_TOKEN_TYPE)
    {
        // 检查第三个元素
        shared_ptr<math_expr_token> mem_index = dynamic_pointer_cast<math_expr_token>(basic_mem_index);
        assert(mem_index != NULL);

        // 递归检查，不能错误
        if (mem_index->static_check() == false)
        {
            cout << "arr_declaration_token::static_check(): invalid mem_index" << endl;
            return false;
        }
    }

    // 通过三个元素的检查
    return true;
}

string arr_declaration_token::get_inited_var_name()
{
    shared_ptr<var_name_token> init_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["mem_ptr_name"]);
    assert(init_var_name->get_token_type() == VAR_NAME_TOKEN_TYPE);
    assert(init_var_name->static_check() == true);

    return init_var_name->get_var_name_str();
}