#include "../kernel_generator.h"

shared_mem_broadcast_token::shared_mem_broadcast_token(vector<shared_ptr<data_type_token>> data_type_of_read_data_arr,
                                                       vector<shared_ptr<var_name_token>> global_mem_read_arr, vector<shared_ptr<var_name_token>> global_mem_read_index_arr,
                                                       vector<shared_ptr<var_name_token>> dest_variable_arr)
    : basic_token(false, SHARED_MEM_BROADCAST_TOKEN_TYPE)
{
    // 四个数组的大小一样，并且一定都大于0
    assert(data_type_of_read_data_arr.size() > 0);
    assert(data_type_of_read_data_arr.size() == global_mem_read_arr.size() && global_mem_read_arr.size() == global_mem_read_index_arr.size() && global_mem_read_index_arr.size() == dest_variable_arr.size());

    // 数组中不能有空指针
    for (int i = 0; i < data_type_of_read_data_arr.size(); i++)
    {
        assert(data_type_of_read_data_arr[i] != NULL);
        assert(global_mem_read_arr[i] != NULL);
        assert(global_mem_read_index_arr[i] != NULL);
        assert(dest_variable_arr[i] != NULL);
    }

    this->broadcast_data_num = data_type_of_read_data_arr.size();

    // 存到child map中
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        this->token_of_child_map["data_type_of_read_data_" + to_string(i)] = data_type_of_read_data_arr[i];
        this->token_of_child_map["global_mem_read_" + to_string(i)] = global_mem_read_arr[i];
        this->token_of_child_map["global_mem_read_index_" + to_string(i)] = global_mem_read_index_arr[i];
        this->token_of_child_map["dest_variable_" + to_string(i)] = dest_variable_arr[i];
    }
}

// 默认构造函数
shared_mem_broadcast_token::shared_mem_broadcast_token()
    : basic_token(false, SHARED_MEM_BROADCAST_TOKEN_TYPE)
{
    // 什么也不做
    this->broadcast_data_num = 0;
}

// 增加一个新的要广播的数据
void shared_mem_broadcast_token::add_broadcast_data(shared_ptr<data_type_token> data_type_of_read_data, shared_ptr<var_name_token> global_mem_read,
                                                    shared_ptr<var_name_token> global_mem_read_index, shared_ptr<var_name_token> dest_variable)
{
    // 不能有空指针
    assert(data_type_of_read_data != NULL);
    assert(global_mem_read != NULL);
    assert(global_mem_read_index != NULL);
    assert(dest_variable != NULL);

    // 要加入的元素在加之前不存在
    assert(this->child_is_exist("data_type_of_read_data_" + to_string(this->broadcast_data_num)) == false);
    assert(this->child_is_exist("global_mem_read_" + to_string(this->broadcast_data_num)) == false);
    assert(this->child_is_exist("global_mem_read_index_" + to_string(this->broadcast_data_num)) == false);
    assert(this->child_is_exist("dest_variable_" + to_string(this->broadcast_data_num)) == false);
    
    this->token_of_child_map["data_type_of_read_data_" + to_string(this->broadcast_data_num)] = data_type_of_read_data;
    this->token_of_child_map["global_mem_read_" + to_string(this->broadcast_data_num)] = global_mem_read;
    this->token_of_child_map["global_mem_read_index_" + to_string(this->broadcast_data_num)] = global_mem_read_index;
    this->token_of_child_map["dest_variable_" + to_string(this->broadcast_data_num)] = dest_variable;

    this->broadcast_data_num = this->broadcast_data_num + 1;
}

string shared_mem_broadcast_token::run()
{
    assert(this->static_check() == true);

    // 首先执行一个全局同步
    string return_str = "__syncthreads();\n";

    // 将需要广播的数据放到shared_mem中，使用第一个线程来处理
    return_str = return_str + "if (threadIdx.x == 0){\n";

    // 将数据一个个拷贝到对应的共享内存中
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        return_str = return_str + this->token_of_child_map["global_mem_read_" + to_string(i)]->run() + "_shared_space[0] = ";
        return_str = return_str + this->token_of_child_map["global_mem_read_" + to_string(i)]->run() + "[" + this->token_of_child_map["global_mem_read_index_" + to_string(i)]->run() + "];\n";
    }

    // 下括号
    return_str = return_str + "}\n";

    // 执行一次全局同步
    return_str = return_str + "__syncthreads();\n";

    // 将shared mem中的数据放到寄存器中
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        // 数据类型，声明和内容的读取应该是分开的
        // return_str = return_str + this->token_of_child_map["data_type_of_read_data_" + to_string(i)]->run() + " ";
        // 输出变量名
        return_str = return_str + this->token_of_child_map["dest_variable_" + to_string(i)]->run() + " = " + this->token_of_child_map["global_mem_read_" + to_string(i)]->run() + "_shared_space[0];\n";
    }

    return return_str;
}

vector<string> shared_mem_broadcast_token::needed_shared_mem_name()
{
    vector<string> return_vec;
    // 遍历所有的共享内存。
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        // 共享内存变量名
        assert(this->child_is_exist("global_mem_read_" + to_string(i)) == true);

        shared_ptr<var_name_token> global_mem_read_arr = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["global_mem_read_" + to_string(i)]);
        assert(global_mem_read_arr->get_token_type() == VAR_NAME_TOKEN_TYPE);

        // 用来广播的共享内存名字
        return_vec.push_back(global_mem_read_arr->run() + "_shared_space");
    }

    return return_vec;
}

vector<unsigned int> shared_mem_broadcast_token::needed_shared_mem_array_size()
{
    vector<unsigned int> return_vec;
    
    // 遍历所有的共享内存。
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        // 所有的共享内存只有1那么大
        return_vec.push_back(1);
    }

    return return_vec;
}

// 遍历所有的数据类型，给出对应每一个数据的数据类型
vector<data_type> shared_mem_broadcast_token::needed_shared_mem_data_type()
{
    vector<data_type> return_vec;

    // 遍历所有共享内存
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        shared_ptr<data_type_token> token_ptr = dynamic_pointer_cast<data_type_token>(this->token_of_child_map["data_type_of_read_data_" + to_string(i)]);
        assert(token_ptr->get_token_type() == DATA_TYPE_TOKEN_TYPE);

        return_vec.push_back(token_ptr->get_data_type());
    }

    return return_vec;
}

bool shared_mem_broadcast_token::static_check()
{
    // 所有共享的内容的检查
    for (int i = 0; i < this->broadcast_data_num; i++)
    {
        // 首先是全局内存中数据的类型必须存在，并且必然不能是指针类型
        assert(this->child_is_exist("data_type_of_read_data_" + to_string(i)) == true);

        shared_ptr<data_type_token> data_type_of_read_data = dynamic_pointer_cast<data_type_token>(this->token_of_child_map["data_type_of_read_data_" + to_string(i)]);
        assert(data_type_of_read_data->get_token_type() == DATA_TYPE_TOKEN_TYPE);

        // 递归检查
        if (data_type_of_read_data->static_check() == false)
        {
            cout << "shared_mem_broadcast_token::static_check(): invalid data_type_of_read_data" << endl;
            return false;
        }

        // 不能是指针
        if (data_type_of_read_data->get_is_pointer() == true)
        {
            cout << "shared_mem_broadcast_token::static_check(): the data type in input array shouldn't be pointer" << endl;
            return false;
        }

        // 全局内存变量
        assert(this->child_is_exist("global_mem_read_" + to_string(i)) == true);

        shared_ptr<var_name_token> global_mem_read = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["global_mem_read_" + to_string(i)]);
        assert(global_mem_read->get_token_type() == VAR_NAME_TOKEN_TYPE);

        // 递归检查
        if (global_mem_read->static_check() == false)
        {
            cout << "shared_mem_broadcast_token::static_check(): invalid global_mem_read" << endl;
            return false;
        }

        // 必须是GLOBAL类型
        if (global_mem_read->get_var_type() != GLOBAL_MEM_VAR_TYPE)
        {
            cout << "shared_mem_broadcast_token::static_check(): global_mem_read is not GLOBAL_MEM_VAR_TYPE" << endl;
            return false;
        }

        // 索引
        assert(this->child_is_exist("global_mem_read_index_" + to_string(i)) == true);

        // 索引必须是REGISTER类型
        shared_ptr<var_name_token> global_mem_read_index = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["global_mem_read_index_" + to_string(i)]);
        assert(global_mem_read_index->get_token_type() == VAR_NAME_TOKEN_TYPE);

        if (global_mem_read_index->static_check() == false)
        {
            cout << "shared_mem_broadcast_token::static_check(): invalid global_mem_read_index" << endl;
            return false;
        }

        // 必须是REGISTER
        if (global_mem_read_index->get_var_type() != REGISTER_VAR_TYPE)
        {
            cout << "shared_mem_broadcast_token::static_check(): global_mem_read_index is not REGISTER_VAR_TYPE" << endl;
            return false;
        }

        // 目标变量
        shared_ptr<var_name_token> dest_variable = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["dest_variable_" + to_string(i)]);
        assert(dest_variable->get_token_type() == VAR_NAME_TOKEN_TYPE);

        if (dest_variable->static_check() == false)
        {
            cout << "shared_mem_broadcast_token::static_check(): invalid dest_variable" << endl;
            return false;
        }

        // 必须是REGISTER
        if (dest_variable->get_var_type() != REGISTER_VAR_TYPE)
        {
            cout << "shared_mem_broadcast_token::static_check(): dest_variable is not REGISTER_VAR_TYPE" << endl;
            return false;
        }
    }

    return true;
}