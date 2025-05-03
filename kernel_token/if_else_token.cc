#include "../kernel_generator.h"

if_else_token::if_else_token(vector<shared_ptr<math_expr_token>> condition, vector<unsigned int> ptr, vector<shared_ptr<var_assign_token>> assign_expr)
    : basic_token(false, IF_ELSE_TOKEN_TYPE)
{
    assert(condition.size() > 0);
    assert(assign_expr.size() > 0);

    for (unsigned int i = 0; i < condition.size(); i++)
    {
        this->token_of_child_map["condition_" + to_string(i)] = condition[i];
    }

    for (unsigned int j = 0; j < assign_expr.size(); j++)
    {
        this->token_of_child_map["assign_" + to_string(j)] = assign_expr[j];
    }
    this->ptr = ptr;
}


// 实际上执行复制，先左后右，最后有分号
string if_else_token::run()
{
    string return_str = "if";
    vector<unsigned int> pointer = this->ptr;
    unsigned int start = 0;
    unsigned int end = 0;
    unsigned int i = 0;
    for (i = 0; i < 100; i++)
    {
        if (this->child_is_exist("condition_" + to_string(i)) == true)
        {
            if (i == 0)
            {
                return_str += "(" + this->get_token_of_child("condition_" + to_string(i))->run() + "){";
            }
            else
            {
                return_str += "else if(" + this->get_token_of_child("condition_" + to_string(i))->run() + "){";
            }

            end += pointer[i];
            for (unsigned int j = start; j < end; j++)
            {
                if (this->child_is_exist("assign_" + to_string(j)) == true)
                {
                    return_str += this->get_token_of_child("assign_" + to_string(i))->run();
                }
                else
                {
                    assert(false);
                }
            }
            start += pointer[i];
            return_str += "}";
        }
        else
        {
            break;
        }
    }
    
    if (pointer.size() == i + 1)
    {
        return_str += "else{";
        end += pointer[i];
        for (unsigned int j = start; j < end; j++)
        {
            if (this->child_is_exist("assign_" + to_string(j)) == true)
            {
                return_str += this->get_token_of_child("assign_" + to_string(i))->run();
            }
            else
            {
                assert(false);
            }
        }
        return_str += "}";
    }

    return return_str;
}

// 执行静态检查
bool if_else_token::static_check()
{
    return true;
}
