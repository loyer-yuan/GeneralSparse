#include "kernel_generator.h"

string convert_token_type_to_string(TOKEN_TYPE token_type)
{
    if (token_type == NONE_TOKEN_TYPE)
    {
        return "NONE_TOKEN_TYPE";
    }

    if (token_type == FOR_TOKEN_TYPE)
    {
        return "FOR_TOKEN_TYPE";
    }

    if (token_type == MATH_EXPR_TOKEN_TYPE)
    {
        return "MATH_EXPR_TOKEN_TYPE";
    }

    if (token_type == VAR_NAME_TOKEN_TYPE)
    {
        return "VAR_NAME_TOKEN_TYPE";
    }

    if (token_type == ARR_ACCESS_TOKEN_TYPE)
    {
        return "ARR_ACCESS_TOKEN_TYPE";
    }

    if (token_type == DATA_TYPE_TOKEN_TYPE)
    {
        return "DATA_TYPE_TOKEN_TYPE";
    }

    if (token_type = VAR_INIT_TOKEN_TYPE)
    {
        return "VAR_INIT_TOKEN_TYPE";
    }

    if (token_type == SHARED_MEM_WRITE_TOKEN_TYPE)
    {
        return "SHARED_MEM_WRITE_TOKEN_TYPE";
    }

    if (token_type == METADATA_GET_BASIC_TOKEN_TYPE)
    {
        return "METADATA_GET_BASIC_TOKEN_TYPE";
    }

    if (token_type == REDUCTION_BASIC_TOKEN_TYPE)
    {
        return "REDUCTION_BASIC_TOKEN_TYPE";
    }

    if (token_type == FOR_BASIC_TOKEN_TYPE)
    {
        return "FOR_BASIC_TOKEN_TYPE";
    }

    if (token_type == FOR_TOKEN_TYPE)
    {
        return "FOR_TOKEN_TYPE";
    }

    if (token_type == SHARED_MEM_BROADCAST_TOKEN_TYPE)
    {
        return "SHARED_MEM_BROADCAST_TOKEN_TYPE";
    }

    if (token_type == VAR_ASSIGN_TOKEN_TYPE)
    {
        return "VAR_ASSIGN_TOKEN_TYPE";
    }

    if (token_type == RAW_CUDA_CODE_LINE)
    {
        return "RAW_CUDA_CODE_LINE";
    }

    if (token_type == METADATA_GET_TOKEN_TYPE)
    {
        return "METADATA_GET_TOKEN_TYPE";
    }

    if (token_type == BASIC_GLUE_CODE)
    {
        return "BASIC_GLUE_CODE";
    }

    cout << "convert_token_type_to_string: invaild token type" << endl;
    assert(false);
    return "";
}

bool check_token_type(TOKEN_TYPE token_type)
{
    if (token_type == NONE_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == FOR_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == MATH_EXPR_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == VAR_NAME_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == ARR_ACCESS_TOKEN_TYPE)
    {
        return true;   
    }

    if (token_type == DATA_TYPE_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == VAR_INIT_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == SHARED_MEM_INIT_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == SHARED_MEM_WRITE_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == FOR_BASIC_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == REDUCTION_BASIC_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == METADATA_GET_BASIC_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == FOR_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == SHARED_MEM_BROADCAST_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == VAR_ASSIGN_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == RAW_CUDA_CODE_LINE)
    {
        return true;
    }

    if (token_type == METADATA_GET_TOKEN_TYPE)
    {
        return true;
    }

    if (token_type == BASIC_GLUE_CODE)
    {
        return true;
    }

    if(token_type == IF_ELSE_TOKEN_TYPE)
    {
        return true;
    }

    if(token_type == ARR_DECLARATION_TOKEN_TYPE)
    {
        return true;
    }

    return false;
}

// 打印当前变量类型
string convert_var_type_to_string(VAR_TYPE var_type)
{
    if (var_type == NONE_VAR_TYPE)
    {
        return "NONE_VAR_TYPE";
    }

    if (var_type == GLOBAL_MEM_VAR_TYPE)
    {
        return "GLOBAL_MEM_VAR_TYPE";
    }

    if (var_type == SHARED_MEM_VAR_TYPE)
    {
        return "SHARED_MEM_VAR_TYPE";
    }

    if (var_type == REGISTER_VAR_TYPE)
    {
        return "REGISTER_VAR_TYPE";
    }

    if (var_type == CONSTANT_VAR_TYPE)
    {
        return "CONSTANT_VAR_TYPE";
    }

    cout << "convert_var_type_to_string: invalid var type" << endl;
    assert(false);
    return "";
}


// 检查变量类型是不是正确
bool check_var_type(VAR_TYPE var_type)
{
    if (var_type == NONE_VAR_TYPE)
    {
        return true;
    }

    if (var_type == GLOBAL_MEM_VAR_TYPE)
    {
        return true;
    }

    if (var_type == SHARED_MEM_VAR_TYPE)
    {
        return true;
    }

    if (var_type == REGISTER_VAR_TYPE)
    {
        return true;
    }

    if (var_type == CONSTANT_VAR_TYPE)
    {
        return true;
    }

    return false;
}

string convert_reduction_token_type_to_string(REDUCTION_TOKEN_TYPE reduction_token_type)
{
    if (reduction_token_type == NONE_REDUCTION_TOKEN_TYPE)
    {
        return "NONE_REDUCTION_TOKEN_TYPE";
    }

    cout << "convert_reduction_token_type_to_string: invalid reduction token type" << endl;
    assert(false);
    return "";
}

bool check_reduction_token_type(REDUCTION_TOKEN_TYPE reduction_token_type)
{
    if (reduction_token_type == NONE_REDUCTION_TOKEN_TYPE)
    {
        return true;        
    }

    return false;
}

string convert_glue_code_token_type_to_string(GLUE_CODE_TOKEN_TYPE glue_code_token_type)
{
    if (glue_code_token_type == NONE_GLUE_CODE_TOKEN_TYPE)
    {
        return "NONE_GLUE_CODE_TOKEN_TYPE";
    }

    cout << "convert_glue_code_token_type_to_string: invalid glue code token type" << endl;
    assert(false);
    return "";
}

bool check_glue_code_token_type(GLUE_CODE_TOKEN_TYPE glue_code_token_type)
{
    if (glue_code_token_type == NONE_GLUE_CODE_TOKEN_TYPE)
    {
        return true;
    }

    return false;
}


basic_token::basic_token(bool is_terminal, TOKEN_TYPE type)
{
    assert(check_token_type(type) == true);
    
    this->is_terminal = is_terminal;
    this->token_type = type;
}

void basic_token::set_token_of_child(string child_token_name, shared_ptr<basic_token> token)
{
    // 之前不存在对应的子节点
    if (this->token_of_child_map.count(child_token_name))
    {
        cout << "basic_token::set_token_of_child: child_token_name has existed, child_token_name:" << child_token_name << endl;
        assert(false);
    }

    // 加入子节点
    this->token_of_child_map[child_token_name] = token;
}

shared_ptr<basic_token> basic_token::get_token_of_child(string child_token_name)
{
    // 之前不存在对应的子节点
    if (this->token_of_child_map.count(child_token_name) == 0)
    {
        cout << "basic_token::get_token_of_child: child_token_name is not existed, child_token_name:" << child_token_name << endl;
        assert(false);
    }

    return token_of_child_map[child_token_name];
}

bool basic_token::child_is_exist(string child_token_name)
{
    if (this->token_of_child_map.count(child_token_name) == 1)
    {
        return true;
    }
    else if (this->token_of_child_map.count(child_token_name) == 0)
    {
        return false;
    }
    else
    {
        cout << "basic_token::child_is_exist: invaild child_token_name, child_token_name:" << child_token_name << endl;
        assert(false);
    }
}

void basic_token::remove_child(string child_token_name)
{
    if (this->token_of_child_map.count(child_token_name))
    {
        cout << "basic_token::remove_child: child_token_name has existed, child_token_name:" << child_token_name << endl;
        assert(false);
    }

    this->token_of_child_map.erase(child_token_name);
}

string var_of_metadata_from_spec_paral(POS_TYPE read_pos_type, string read_metadata_name, int read_sub_matrix_id, shared_ptr<math_expr_token> index_expr)
{
    assert(check_pos_type(read_pos_type) == true);
    assert(index_expr != NULL);
    assert(index_expr->static_check() == true);

    // 将表达式中的内容读出来，字母和数字不变，符号替换
    string index_expr_str = index_expr->run();
    
    string sub_str_for_idx = "";

    // 遍历所有的读取索引相关的内容，使用ASC码来标定唯一的
    for (int i = 0; i < index_expr_str.size(); i++)
    {
        // 如果是数字或者字母就直接写入
        if (isalnum(index_expr_str[i]) != 0)
        {
            sub_str_for_idx = sub_str_for_idx + to_string(index_expr_str[i]);
        }

        // 如果是其他符号，加减乘除余，需要转化为字母asmdr
        if (index_expr_str[i] == '+')
        {
            sub_str_for_idx = sub_str_for_idx + to_string('p');
        }

        if (index_expr_str[i] == '-')
        {
            sub_str_for_idx = sub_str_for_idx + to_string('s');
        }

        if (index_expr_str[i] == '*')
        {
            sub_str_for_idx = sub_str_for_idx + to_string('m');
        }

        if (index_expr_str[i] == '/')
        {
            sub_str_for_idx = sub_str_for_idx + to_string('d');
        }

        if (index_expr_str[i] == '%')
        {
            sub_str_for_idx = sub_str_for_idx + to_string('r');
        }
    }

    // 将位置，metadata名字，子矩阵号合在一起，尾部加上一个item作为变量
    // string return_str = convert_pos_type_to_string(read_pos_type) + "_";
    // return_str = return_str + read_metadata_name + "_" + to_string(read_sub_matrix_id) + "_" + sub_str_for_idx + "_item";
    string return_str = get_metadata_item_name(read_pos_type, read_metadata_name, read_sub_matrix_id) + "_" + sub_str_for_idx + "_item";

    return return_str;
}



string code_of_data_type(data_type type)
{
    if (type == CHAR)
    {
        return "char";
    }
    else if(type == CHAR2)
    {
        return "char2";
    }
    else if (type == CHAR4)
    {
        return "char4";
    }
    else if (type == CHAR8)
    {
        return "char8";
    }
    else if (type == UNSIGNED_CHAR)
    {
        return "unsigned char";
    }
    else if (type == SHORT)
    {
        return "short";
    }
    else if (type == SHORT2)
    {
        return "short2";
    }
    else if (type == SHORT4)
    {
        return "short4";
    }
    else if (type == SHORT8)
    {
        return "short8";
    }
    else if (type == UNSIGNED_SHORT)
    {
        return "unsigned short";
    }
    else if (type == INT)
    {
        return "int";
    }
    else if (type == INT2)
    {
        return "int2";
    }    
    else if (type == INT4)
    {
        return "int4";
    }
    else if (type == UNSIGNED_INT)
    {
        return "unsigned int";
    }
    else if (type == LONG)
    {
        return "long";
    }
    else if (type == LONG2)
    {
        return "long2";
    }
    else if (type == UNSIGNED_LONG)
    {
        return "unsigned long";
    }
    else if (type == LONG_LONG)
    {
        return "long long";
    }
    else if (type == UNSIGNED_LONG_LONG)
    {
        return "unsigned long long";
    }
    else if (type == FLOAT)
    {
        return "float";
    }
    else if (type == FLOAT2)
    {
        return "float2";
    }
    else if (type == FLOAT4)
    {
        return "float4";
    }
    else if (type == DOUBLE)
    {
        return "double";
    }
    else if (type == DOUBLE2)
    {
        return "double2";
    }
    else if (type == BOOL)
    {
        return "bool";
    }
    else if (type == HALF)
    {
        return "half";
    }
    else if (type == HALF2)
    {
        return "half2";
    }
    else if (type == HALF4)
    {
        return "half4";
    }
    else if (type == HALF8)
    {
        return "half8";
    }

    assert(false);
}

void write_string_to_file(string file_name, string output_str)
{
    ofstream outfile(file_name, ios::trunc);
    outfile << output_str << endl;
}

