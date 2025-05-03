#include "../kernel_generator.h"

var_init_token::var_init_token(shared_ptr<data_type_token> data_type_declare, shared_ptr<var_name_token> init_var_name, shared_ptr<math_expr_token> init_math_express)
:basic_token(false, VAR_INIT_TOKEN_TYPE)
{
    assert(data_type_declare != NULL && init_var_name != NULL);
    
    // 将子节点放在对应的内容中，如果是空的就不放了
    this->token_of_child_map["data_type_declare"] = data_type_declare;
    this->token_of_child_map["init_var_name"] = init_var_name;

    if (init_math_express != NULL)
    {
        this->token_of_child_map["init_math_express"] = init_math_express;       
    }
}

string var_init_token::run()
{
    assert(this->child_is_exist("data_type_declare") == true);
    assert(this->child_is_exist("init_var_name") == true);
    assert(this->static_check() == true);
    
    string return_str = "";
    return_str += this->token_of_child_map["data_type_declare"]->run() + " " + this->token_of_child_map["init_var_name"]->run();
    
    // 如果存在初始化，就加入初始化的数学表达式
    if (this->child_is_exist("init_math_express") == true)
    {
        return_str = return_str + " = " + this->token_of_child_map["init_math_express"]->run();
    }

    return_str = return_str + ";";
    
    return return_str;
}

bool var_init_token::static_check()
{
    assert(this->child_is_exist("data_type_declare") == true);
    assert(this->child_is_exist("init_var_name") == true);
    
    // 数据类型声明是必然存在的，进行检查
    shared_ptr<data_type_token> data_type_declare = dynamic_pointer_cast<data_type_token>(this->token_of_child_map["data_type_declare"]);
    assert(data_type_declare->get_token_type() == DATA_TYPE_TOKEN_TYPE);
    
    if (data_type_declare->static_check() == false)
    {
        cout << "var_init_token::static_check(): invaild data_type_declare" << endl;
        return false;
    }

    shared_ptr<var_name_token> init_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["init_var_name"]);
    assert(init_var_name->get_token_type() == VAR_NAME_TOKEN_TYPE);
    
    if (init_var_name->static_check() == false)
    {
        cout << "var_init_token::static_check(): invaild init_var_name" << endl;
        return false;
    }

    if (init_var_name->get_var_type() != REGISTER_VAR_TYPE)
    {
        cout << "var_init_type_token::static_check(): init_var_name is not REGISTER type" << endl;
        return false;
    }

    // 如果存在，计算表达式，给出表达式的静态检查
    if (this->child_is_exist("init_math_express") == true)
    {
        shared_ptr<math_expr_token> init_math_express = dynamic_pointer_cast<math_expr_token>(this->token_of_child_map["init_math_express"]);
        assert(init_math_express->get_token_type() == MATH_EXPR_TOKEN_TYPE);

        if (init_math_express->static_check() == false)
        {
            cout << "var_init_token::static_check(): invaild init_math_express" << endl;
            return false;
        }
    }

    return true;
}

string var_init_token::get_inited_var_name()
{
    assert(this->child_is_exist("init_var_name") == true);

    shared_ptr<var_name_token> init_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["init_var_name"]);
    assert(init_var_name->get_token_type() == VAR_NAME_TOKEN_TYPE);
    assert(init_var_name->static_check() == true);

    return init_var_name->get_var_name_str();
}