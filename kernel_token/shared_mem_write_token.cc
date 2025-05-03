#include "../kernel_generator.h"

shared_mem_write_token::shared_mem_write_token(shared_ptr<var_name_token> shared_mem_name, shared_ptr<var_name_token> input_index, shared_ptr<var_name_token> written_value)
:basic_token(false, SHARED_MEM_WRITE_TOKEN_TYPE)
{
    assert(shared_mem_name != NULL);
    assert(input_index != NULL);
    assert(written_value != NULL);

    this->token_of_child_map["shared_mem_name"] = shared_mem_name;
    this->token_of_child_map["input_index"] = input_index;
    this->token_of_child_map["written_value"] = written_value;
}

// 执行对应函数
string shared_mem_write_token::run()
{
    assert(this->static_check() == true);

    string return_str = this->token_of_child_map["shared_mem_name"]->run() + "[" + this->token_of_child_map["input_index"]->run() + "]";
    return_str = return_str + " = " + this->token_of_child_map["written_value"]->run();
    
    return return_str;
}

// 执行静态检查
bool shared_mem_write_token::static_check()
{
    // 验证共享内存名
    shared_ptr<var_name_token> shared_mem_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["shared_mem_name"]);
    assert(shared_mem_name->get_token_type() == VAR_NAME_TOKEN_TYPE);

    if (shared_mem_name->static_check() == false)
    {
        cout << "shared_mem_write_token::static_check(): invaild shared_mem_name" << endl;
        return false;
    }

    if (shared_mem_name->get_var_type() != SHARED_MEM_VAR_TYPE)
    {
        cout << "shared_mem_write_token::static_check(): shared_mem_name is not SHARED_MEM" << endl;
        return false;
    }

    // 写入位置的索引名
    shared_ptr<var_name_token> input_index = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["input_index"]);
    assert(input_index->get_token_type() == VAR_NAME_TOKEN_TYPE);

    if (input_index->static_check() == false)
    {
        cout << "shared_mem_write_token::static_check(): invalid input_index" << endl;
        return false;
    }

    // 索引名必须是常量和寄存器变量
    if (input_index->get_var_type() != REGISTER_VAR_TYPE && input_index->get_var_type() != CONSTANT_VAR_TYPE)
    {
        cout << "shared_mem_write_token::static_check(): input_index is not REGISTER or CONSTANT" << endl;
        return false;
    }

    // 写入数据，常量或者寄存器变量
    shared_ptr<var_name_token> written_value = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["written_value"]);
    assert(written_value->get_token_type() == VAR_NAME_TOKEN_TYPE);

    if (written_value->static_check() == false)
    {
        cout << "shared_mem_write_token::static_check(): invalid written_value" << endl;
        return false;
    }

    // 索引名必须是常量和寄存器变量
    if (written_value->get_var_type() != REGISTER_VAR_TYPE && written_value->get_var_type() != CONSTANT_VAR_TYPE)
    {
        cout << "shared_mem_write_token::static_check(): written_value is not REGISTER or CONSTANT" << endl;
        return false;
    }

    return true;
}