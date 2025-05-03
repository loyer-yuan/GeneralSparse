#include "../kernel_generator.h"

basic_glue_code::basic_glue_code(shared_ptr<basic_IO_of_reduction> input_IO, shared_ptr<basic_IO_of_reduction> output_IO)
:basic_token(true, BASIC_GLUE_CODE)
{
    // 记录所有的输入
    assert(input_IO != NULL);

    // 拷贝所有的输入
    for (int i = 0; i < input_IO->number_of_unbound_var_num(); i++)
    {
        assert(input_IO->get_unbound_var(i) != NULL);
        this->input_var_vec.push_back(input_IO->get_unbound_var(i));
    }

    if(output_IO == NULL)
    {
        return;
    }

    // 拷贝所有的输出
    for (int i = 0; i < output_IO->number_of_unbound_var_num(); i++)
    {
        assert(output_IO->get_unbound_var(i) != NULL);
        this->output_var_vec.push_back(output_IO->get_unbound_var(i));
    }
}

string basic_glue_code::run()
{
    // 这里仅仅打印元数据，不生成任何实际的代码
    assert(this->static_check() == true);
    
    string return_str = "\ngule:\n{\n";
    
    for (int i = 0; i < this->input_var_vec.size(); i++)
    {
        return_str = return_str + "input_" + to_string(i) + ":" + this->input_var_vec[i]->run() + "\n";
    }

    for (int i = 0; i < this->output_var_vec.size(); i++)
    {
        return_str = return_str + "output_" + to_string(i) + ":" + this->output_var_vec[i]->run() + "\n";
    }

    return_str = return_str + "}\n";

    return return_str;
}

// TODO检查里面的变量名
bool basic_glue_code::static_check()
{
    // 主要看输入输出是不是合法
    // 遍历所有输入变量
    for (int i = 0; i < this->input_var_vec.size(); i++)
    {
        // 当前是不是空指针
        if (this->input_var_vec[i] == NULL)
        {
            cout << "basic_glue_code::static_check: empty input pointer error" << endl;
            return false;
        }
        else
        {
            // 不是空指针，那就检查变量本身是不是有问题
            if (this->input_var_vec[i]->static_check() == false)
            {
                cout << "basic_glue_code::static_check: illegal var name:" << i << endl;
                return false;
            }
        }
    }

    // 遍历所有的输出变量
    for (int i = 0; i < this->output_var_vec.size(); i++)
    {
        // 当前是不是空指针
        if (this->output_var_vec[i] == NULL)
        {
            cout << "basic_glue_code::static_check: empty output pointer error" << endl;
            return false;
        }
        else
        {
            // 不是空指针，那就检查变量本身是不是有问题
            if (this->output_var_vec[i]->static_check() == false)
            {
                cout << "basic_glue_code::static_check: illegal var name:" << i << endl;
                return false;
            }
        }
    }

    return true;
}