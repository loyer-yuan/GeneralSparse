#include "../kernel_generator.h"
#include <cctype>

var_name_token::var_name_token(string var_str, VAR_TYPE var_type)
:basic_token(true, VAR_NAME_TOKEN_TYPE)
{
    assert(var_str != "");
    assert(check_var_type(var_type) == true);
    this->var_str = var_str;
    this->var_type = var_type;
}

string var_name_token::run()
{
    assert(this->static_check() == true);

    // 直接打印变量名
    return this->var_str;
}

bool var_name_token::static_check()
{
    // 检查，对于reigster、global、shared的变量来说，开头不能有数字，剩下的部分只能由大小写字母、下划线、和数字构成
    if (this->var_type == REGISTER_VAR_TYPE || this->var_type == GLOBAL_MEM_VAR_TYPE || this->var_type == SHARED_MEM_VAR_TYPE)
    {
        for (int i = 0; i < this->var_str.size(); i++)
        {
            if (i == 0)
            {
                // 第一位必须是字母
                if (isalpha(this->var_str[i]) == 0)
                {
                    cout << "var_name_token::static_check(): the first character is not letter" << endl;
                    cout << "var_name_token::static_check(): string of name:" << this->var_str << endl;
                    return false;
                }
            }
            else
            {
                // 剩下的必须是下划线数字字母
                if (isalnum(this->var_str[i]) == 0 && this->var_str[i] != '_')
                {
                    cout << "var_name_token::static_check(): invaild remain characters" << endl;
                    return false;
                }
            }
        }
    }
    else if (this->var_type == CONSTANT_VAR_TYPE)
    {
        // 第一位是数字，剩下的是点或者数字
        for (int i = 0; i < this->var_str.size(); i++)
        {
            if (i == 0)
            {
                // 第一位必须是字母
                if (isdigit(this->var_str[i]) == 0)
                {
                    cout << "var_name_token::static_check(): the first character is not number" << endl;
                    return false;
                }
            }
            else
            {
                // 剩下的必须是数字和小数点
                if (isdigit(this->var_str[i]) == 0 && this->var_str[i] != '.')
                {
                    cout << "var_name_token::static_check(): invaild remain characters" << endl;
                    return false;
                }
            }
        }
    }
    else
    {
        assert(false);
    }

    // 通过所有验证
    return true;
}