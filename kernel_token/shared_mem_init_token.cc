#include "../kernel_generator.h"

shared_mem_init_token::shared_mem_init_token(shared_ptr<data_type_token> data_type_declare, shared_ptr<var_name_token> init_shared_mem_var_name, shared_ptr<var_name_token> shared_mem_size_var_name)
:basic_token(false, SHARED_MEM_INIT_TOKEN_TYPE)
{
    assert(data_type_declare != NULL);
    assert(init_shared_mem_var_name != NULL);
    assert(shared_mem_size_var_name != NULL);

    this->token_of_child_map["data_type_declare"] = data_type_declare;
    this->token_of_child_map["init_shared_mem_var_name"] = init_shared_mem_var_name;
    this->token_of_child_map["shared_mem_size_var_name"] = shared_mem_size_var_name;
}

string shared_mem_init_token::run()
{
    assert(this->static_check() == true);

    string return_str = "__shared__ ";
    return_str = return_str + this->token_of_child_map["data_type_declare"]->run() + " ";
    return_str = return_str + this->token_of_child_map["init_shared_mem_var_name"]->run() + "[";
    return_str = return_str + this->token_of_child_map["shared_mem_size_var_name"]->run() + "]";

    return return_str;
}

// 静态检查，分别检查三个子节点的类型，并且对三个子节点执行递归的静态检查
bool shared_mem_init_token::static_check()
{
    // 类型名检查
    shared_ptr<data_type_token> data_type_declare = dynamic_pointer_cast<data_type_token>(this->token_of_child_map["data_type_declare"]);
    assert(data_type_declare->get_token_type() == DATA_TYPE_TOKEN_TYPE);

    // 静态检查
    if (data_type_declare->static_check() == false)
    {
        cout << "shared_mem_init_token::static_check(): invalid data_type_declare" << endl;
        return false;
    }

    // 共享内存变量的检查
    shared_ptr<var_name_token> init_shared_mem_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["init_shared_mem_var_name"]);
    assert(init_shared_mem_var_name->get_token_type() == VAR_NAME_TOKEN_TYPE);

    // 静态检查
    if (init_shared_mem_var_name->static_check() == false)
    {
        cout << "shared_mem_init_token::static_check(): invalid init_shared_mem_var_name" << endl;
        return false;
    }

    // 检查类型，必须是共享内存
    if (init_shared_mem_var_name->get_var_type() != SHARED_MEM_VAR_TYPE)
    {
        cout << "shared_mem_init_token::static_check(): init_shared_mem_var_name is not SAHRED_MEM_TYPE" << endl;
        return false;
    }

    // 共享内存数组的大小必须是常量类型
    shared_ptr<var_name_token> shared_mem_size_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["shared_mem_size_var_name"]);
    assert(shared_mem_size_var_name->get_token_type() == VAR_NAME_TOKEN_TYPE);

    // 静态检查
    if (shared_mem_size_var_name->static_check() == false)
    {
        cout << "shared_mem_init_token::static_check(): invalid shared_mem_size_var_name" << endl;
        return false;
    }

    // 类型必须是常量类型
    if (shared_mem_size_var_name->get_var_type() != CONSTANT_VAR_TYPE)
    {
        cout << "shared_mem_init_token::static_check(): shared_mem_size_var_name is not CONSTANT" << endl;
        return false;
    }
    
    return true;
}