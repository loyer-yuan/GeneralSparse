#include "../kernel_generator.h"

arr_access_token::arr_access_token(shared_ptr<var_name_token> dest_var_name, shared_ptr<var_name_token> mem_ptr_name, shared_ptr<var_name_token> mem_index)
    : basic_token(false, ARR_ACCESS_TOKEN_TYPE)
{
    assert(dest_var_name != NULL);
    assert(mem_ptr_name != NULL);
    assert(mem_index != NULL);

    // 将子节点放到对应的内容中
    this->token_of_child_map["mem_ptr_name"] = mem_ptr_name;
    this->token_of_child_map["dest_var_name"] = dest_var_name;
    this->token_of_child_map["mem_index"] = mem_index;
}

arr_access_token::arr_access_token(shared_ptr<var_name_token> dest_var_name, shared_ptr<var_name_token> mem_ptr_name, string mem_index)
    : basic_token(false, ARR_ACCESS_TOKEN_TYPE)
{
    assert(dest_var_name != NULL);
    assert(mem_ptr_name != NULL);

    // 将子节点放到对应的内容中
    this->token_of_child_map["mem_ptr_name"] = mem_ptr_name;
    this->token_of_child_map["dest_var_name"] = dest_var_name;
    this->mem_index = mem_index;
}

string arr_access_token::run()
{
    assert(this->child_is_exist("dest_var_name") == true);
    assert(this->child_is_exist("mem_ptr_name") == true);
    assert(this->static_check() == true);
    string index;
    if(this->child_is_exist("mem_index") == true)
    {
        index = this->token_of_child_map["mem_index"]->run();
    }
    else
    {
        index = this->mem_index;
    }

    return this->token_of_child_map["dest_var_name"]->run() + " = " + this->token_of_child_map["mem_ptr_name"]->run() + "[" + index + "];";
}

bool arr_access_token::static_check()
{
    assert(this->child_is_exist("dest_var_name") == true);
    assert(this->child_is_exist("mem_ptr_name") == true);

  
    shared_ptr<var_name_token> dest_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["dest_var_name"]);
    assert(dest_var_name != NULL);

    // 第一个变量是REGISTER类型的
    if (dest_var_name->get_var_type() != REGISTER_VAR_TYPE)
    {
        cout << "arr_access_token::static_check(): dest_var_name type is not REGISTER" << endl;
        return false;
    }

    if (dest_var_name->static_check() == false)
    {
        cout << "arr_access_token::static_check(): invalid dest_var_name" << endl;
        return false;
    }

    // 检查第二个元素
    shared_ptr<var_name_token> mem_ptr_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["mem_ptr_name"]);
    assert(mem_ptr_name != NULL);

    // 第二个变量是GLOBAL和SHARED类型，如果
    if (!(mem_ptr_name->get_var_type() == SHARED_MEM_VAR_TYPE || mem_ptr_name->get_var_type() == GLOBAL_MEM_VAR_TYPE) || mem_ptr_name->static_check() == false)
    {
        cout << "arr_access_token::static_check(): mem_ptr_name is not SHARED_MEM and GLOBAL_MEM" << endl;
        return false;
    }
    return true;
}