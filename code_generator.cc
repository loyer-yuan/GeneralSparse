#include "code_generator.hpp"
#include "config.hpp"

void code_generator::add_new_metadata_dependency(POS_TYPE pos, string real_name, int sub_matrix_id)
{
    // cout << "code_generator::add_new_metadata_dependency: here" << endl;
    assert(check_pos_type(pos) == true);
    assert(sub_matrix_id >= 0);
    assert(this->meta_data_set_ptr->check() == true);
    assert(this->check() == true);
    assert(this->sub_matrix_id == sub_matrix_id);

    // 添加的索引是当前不存在的
    // string meta_item_name = convert_pos_type_to_string(pos) + "_" + real_name + "_" + to_string(sub_matrix_id);

    if (this->if_linear_compress(pos, real_name, sub_matrix_id) == true)
    {
        return;
    }

    if (this->if_branch_compress(pos, real_name, sub_matrix_id) == true)
    {
        return;
    }

    if (this->if_cycle_linear_compress(pos, real_name, sub_matrix_id) == true)
    {
        return;
    }

    if (this->if_cycle_increase_compress(pos, real_name, sub_matrix_id) == true)
    {
        return;
    }

    if (this->if_residual_compress(pos, real_name, sub_matrix_id) == true)
    {
        real_name = real_name + "_res";
        string meta_item_name = get_metadata_item_name(pos, real_name, sub_matrix_id);
        if(this->added_metadata_dependency.count(meta_item_name) != 0)
        {
            return;
        }

        assert(this->meta_data_set_ptr->is_exist(pos, real_name, sub_matrix_id) == true);

        this->pos_of_needed_metadata_vec.push_back(pos);
        this->real_name_of_needed_metadata_vec.push_back(real_name);
        this->sub_matrix_id_of_needed_metadata_vec.push_back(sub_matrix_id);
        this->added_metadata_dependency.insert(meta_item_name);

        return;
    }

    string meta_item_name = get_metadata_item_name(pos, real_name, sub_matrix_id);

    if(this->added_metadata_dependency.count(meta_item_name) != 0)
    {
        return;
    }
    assert(this->meta_data_set_ptr->is_exist(pos, real_name, sub_matrix_id) == true);

    // 插入三个数据
    this->pos_of_needed_metadata_vec.push_back(pos);
    this->real_name_of_needed_metadata_vec.push_back(real_name);
    this->sub_matrix_id_of_needed_metadata_vec.push_back(sub_matrix_id);
    // this->memory_access_fusing_op_flag_vec.push_back(mem_access_fusion_op);

    // 插入一个加入对应数据的记录
    this->added_metadata_dependency.insert(meta_item_name);
}

void code_generator::add_new_fused_metadata_access(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<math_expr_token> metadata_access_index_expr)
{
    // 增加一个合并之后的访问
    assert(check_pos_type(pos) == true);
    assert(sub_matrix_id >= 0);
    assert(this->meta_data_set_ptr->check() == true);
    assert(this->check() == true);
    assert(this->sub_matrix_id == sub_matrix_id);
    assert(metadata_access_index_expr != NULL);
    assert(metadata_access_index_expr->static_check() == true);

    // 之前存在对应的合并访问则直接返回
    if(this->if_fused_metadata_access_exist(pos, real_name, sub_matrix_id, metadata_access_index_expr) == true)
    {
        return;
    }

    // 将内容登记到对应的数组中
    // 内容在元数据库中存在
    assert(this->meta_data_set_ptr->is_exist(pos, real_name, sub_matrix_id) == true);

    // 添加四个数据
    this->pos_of_fused_metadata_vec.push_back(pos);
    this->real_name_of_fused_metadata_vec.push_back(real_name);
    this->sub_matrix_id_of_fused_metadata_vec.push_back(sub_matrix_id);
    this->access_index_fused_metadata_vec.push_back(metadata_access_index_expr);
}

bool code_generator::check()
{
    // 递归检查metadata set
    if (this->meta_data_set_ptr == NULL)
    {
        cout << "code_generator::check(): this->meta_data_set_ptr is an empty pointer" << endl;
        return false;
    }

    if (this->meta_data_set_ptr->check() == false)
    {
        cout << "code_generator::check(): error in this->meta_data_set_ptr" << endl;
        return false;
    }

    // 子矩阵号要合法
    if (this->sub_matrix_id < 0)
    {
        cout << "code_generator::check(): illegal sub_matrix_id: " << this->sub_matrix_id << endl;
        return false;
    }
    else
    {
        // 查看当前子矩阵号是不是真的存在，主要看行列值是不是同时存在
        if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->sub_matrix_id) == false ||
            this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->sub_matrix_id) == false ||
            this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->sub_matrix_id) == false)
        {
            cout << "code_generator::check(): corresponding sub-matrix is not existing, id:" << this->sub_matrix_id << endl;
            return false;
        }
    }

    // 数据依赖三个数组的大小和set的大小是一致的
    if (this->pos_of_needed_metadata_vec.size() != this->real_name_of_needed_metadata_vec.size() ||
        this->real_name_of_needed_metadata_vec.size() != this->sub_matrix_id_of_needed_metadata_vec.size() ||
        this->sub_matrix_id_of_needed_metadata_vec.size() != this->added_metadata_dependency.size())
    // || this->added_metadata_dependency.size() != this->memory_access_fusing_op_flag_vec.size())
    {
        cout << "code_generator::check(): the record nums of added metadata item are not matched" << endl;
        return false;
    }

    // 合并访问的四个数组的大小是一致的
    if (this->pos_of_fused_metadata_vec.size() != this->real_name_of_fused_metadata_vec.size() ||
        this->sub_matrix_id_of_fused_metadata_vec.size() != this->real_name_of_fused_metadata_vec.size() ||
        this->real_name_of_fused_metadata_vec.size() != this->access_index_fused_metadata_vec.size())
    {
        cout << "code_generator::check(): the record nums of fused metadata access are not matched" << endl;
        return false;
    }

    // 登记的元数据所属的子矩阵是一样的
    for (int i = 0; i < this->sub_matrix_id_of_needed_metadata_vec.size(); i++)
    {
        if (this->sub_matrix_id_of_needed_metadata_vec[i] != this->sub_matrix_id)
        {
            cout << "code_generator::check(): the sub-matrix id is not match" << endl;
            return false;
        }
    }

    // 遍历所有的元数据依赖，都分别要满足要求
    for (int i = 0; i < this->sub_matrix_id_of_needed_metadata_vec.size(); i++)
    {
        // 查看元数据的位置，名字和子矩阵号
        POS_TYPE metadata_item_pos = this->pos_of_needed_metadata_vec[i];
        string metadata_item_real_name = this->real_name_of_needed_metadata_vec[i];
        int metadata_item_sub_matrix_id = this->sub_matrix_id_of_needed_metadata_vec[i];

        if (check_pos_type(metadata_item_pos) == false)
        {
            cout << "code_generator::check(): pos of needed metadata item " << i << " is illegal" << endl;
            return false;
        }

        // 内容是存在
        if (this->meta_data_set_ptr->is_exist(metadata_item_pos, metadata_item_real_name, metadata_item_sub_matrix_id) == false)
        {
            cout << "code_generator::check(): needed metadata item " << i << " is not in the metadata set" << endl;
            return false;
        }
    }

    for (int i = 0; i < this->sub_matrix_id_of_fused_metadata_vec.size(); i++)
    {
        POS_TYPE metadata_item_pos = this->pos_of_fused_metadata_vec[i];
        string metadata_item_real_name = this->real_name_of_fused_metadata_vec[i];
        int metadata_item_sub_matrix_id = this->sub_matrix_id_of_fused_metadata_vec[i];
        shared_ptr<math_expr_token> metadata_item_access_index = this->access_index_fused_metadata_vec[i];

        if (check_pos_type(metadata_item_pos) == false)
        {
            cout << "code_generator::check(): pos of fused metadata item " << i << " is illegal" << endl;
            return false;
        }

        // 内容是存在
        if (this->meta_data_set_ptr->is_exist(metadata_item_pos, metadata_item_real_name, metadata_item_sub_matrix_id) == false)
        {
            cout << "code_generator::check(): fused metadata item " << i << " is not in the metadata set" << endl;
            return false;
        }

        // 索引对应的表达式满足要求
        if (metadata_item_access_index->static_check() == false)
        {
            cout << "code_generator::check(): metadata access expr of fused metadata item " << i << " is illegal" << endl;
            return false;
        }

        // // 内容在needed metadata记录中也是有的
        // if (this->if_dependency_exist(metadata_item_pos, metadata_item_real_name, metadata_item_sub_matrix_id) == false)
        // {
        //     cout << "code_generator::check(): fused metadata item " << i << " is not recorded in needed metadata records" << endl;
        //     return false;
        // }
    }

    // 共享内存的几个位置应当相等
    if (this->needed_shared_mem_data_type_vec.size() != this->needed_shared_mem_name_vec.size() ||
        this->needed_shared_mem_name_vec.size() != this->needed_shared_mem_size_vec.size() ||
        this->needed_shared_mem_size_vec.size() != this->needed_shared_mem_data_type_vec.size())
    {
        cout << "code_generator::check(): vectors of shared memory info are not in the same size" << endl;
        return false;
    }

    return true;
}

bool code_generator::if_dependency_exist(POS_TYPE pos, string real_name, int sub_matrix_id)
{
    // cout << "code_generator::if_dependency_exist: here" << endl;
    // 执行当前内部的检查
    // 这里不能加检查，防止递归
    // assert(this->check() == true);
    assert(check_pos_type(pos) == true);
    assert(sub_matrix_id == this->sub_matrix_id);

    // 查看对应的依赖是不是存在
    string meta_item_name = get_metadata_item_name(pos, real_name, sub_matrix_id);

    // convert_pos_type_to_string(pos) + "_" + real_name + "_" + to_string(sub_matrix_id);

    if (this->added_metadata_dependency.count(meta_item_name) == 1)
    {
        return true;
    }

    return false;
}

bool code_generator::if_fused_metadata_access_exist(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<math_expr_token> metadata_access_index_expr)
{
    // 执行当前内部的检查
    assert(this->check() == true);
    assert(check_pos_type(pos) == true);
    assert(sub_matrix_id == this->sub_matrix_id);
    assert(metadata_access_index_expr->static_check() == true);

    // 查看是不是存在对应的合并访问
    // 遍历所有的合并访问记录
    for (unsigned long i = 0; i < this->pos_of_fused_metadata_vec.size(); i++)
    {
        // 分别查看位置，名字，子矩阵号和访问的索引表达式
        POS_TYPE metadata_item_pos = this->pos_of_fused_metadata_vec[i];
        string metadata_item_real_name = this->real_name_of_fused_metadata_vec[i];
        int metadata_item_sub_matrix_id = this->sub_matrix_id_of_fused_metadata_vec[i];
        shared_ptr<math_expr_token> metadata_item_access_index = this->access_index_fused_metadata_vec[i];

        // 查看和输入是不是一样的
        if (metadata_item_pos == pos && metadata_item_real_name == real_name)
        {
            if (metadata_item_sub_matrix_id == sub_matrix_id && (metadata_item_access_index == metadata_access_index_expr || metadata_item_access_index->run() == metadata_access_index_expr->run()))
            {
                return true;
            }
        }
    }

    return false;
}

string code_generator::generate_matrix_format_read_code()
{
    stringstream format_read_code;
    assert(this->check());
    unsigned long K = get_config()["DENSE_MATRIX_SIZE"].as_integer();

    // 在CPU端读入数据，指针的名字就是Metadata set中名字
    for (unsigned long i = 0; i < this->pos_of_needed_metadata_vec.size(); i++)
    {


        POS_TYPE pos_of_needed_index_array = this->pos_of_needed_metadata_vec[i];
        string real_name_of_needed_index_array = this->real_name_of_needed_metadata_vec[i];
        int sub_matrix_id_of_needed_index_array = this->sub_matrix_id_of_needed_metadata_vec[i];

        assert(sub_matrix_id_of_needed_index_array == this->sub_matrix_id);

        // 格式索引文件的前缀
        string format_index_file_prefix = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(this->output_id);

        // 当前数据在metadata set中必然存在
        assert(this->meta_data_set_ptr->is_exist(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array));
        // 获得对应的通用数组指针
        shared_ptr<universal_array> index_array_ptr = this->meta_data_set_ptr
                                                          ->get_element(pos_of_needed_index_array,
                                                                        real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array)
                                                          ->get_metadata_arr();

        data_type data_type_of_index = index_array_ptr->get_compress_data_type();

        // // 这里可能存在数据类型压缩，主要是将data_type_of_index压缩成更小的数据类型，浮点类型不参与
        // if (get_config()["DATA_TYPE_COMPRESS"].as_bool() == true)
        // {
        //     cout << "code_generator::generate_matrix_format_read_code: data type compression is not supported" << endl;
        //     assert(false);
        // }

        format_read_code << code_of_data_type(data_type_of_index) << "* ";
        if (get_config()["HALF"].as_bool() == true && data_type_of_index == FLOAT)
        {
            format_read_code << get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array) << "_ = ";
        }
        else
        {
            format_read_code << get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array) << " = ";
        }
        // 强制类型转换调用读索引的函数
        format_read_code << "(" << code_of_data_type(data_type_of_index) << " *)read_arr_from_file_with_data_type(";
        // 读索引的函数的形参，索引的长度，索引的数据类型的枚举
        format_read_code << index_array_ptr->get_len() << "," << convert_data_type_to_string(data_type_of_index) << ",";
        // 然后是对应文件的文件名
        format_read_code << "\"" << format_index_file_prefix << "/" << get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array) << "\");" << endl;

        if(get_config()["HALF"].as_bool() == true && data_type_of_index == FLOAT)
        {
            format_read_code << code_of_data_type(HALF) << "* ";
            format_read_code << get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array) << " = (half *)malloc(sizeof(half) * " << index_array_ptr->get_len() << ");" << endl;
            format_read_code << "for (unsigned long i = 0; i < " << index_array_ptr->get_len() << "; i++){" << endl;
            format_read_code << get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array) << "[i] = ";
            format_read_code << "__float2half(" << get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array) << "_[i]);" << endl;
            format_read_code << "}" << endl;
        }
    }

    format_read_code << endl;

    format_read_code << "cudaSetDevice(" << get_config()["DEFAULT_DEVICE_ID"] << ");" << endl;
    // 拷贝到指向设备的数组
    for (unsigned long i = 0; i < this->pos_of_needed_metadata_vec.size(); i++)
    {


        POS_TYPE pos_of_needed_index_array = this->pos_of_needed_metadata_vec[i];
        string real_name_of_needed_index_array = this->real_name_of_needed_metadata_vec[i];
        int sub_matrix_id_of_needed_index_array = this->sub_matrix_id_of_needed_metadata_vec[i];

        assert(sub_matrix_id_of_needed_index_array == this->sub_matrix_id);

        // 当前数据在metadata set中必然存在
        assert(this->meta_data_set_ptr->is_exist(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array));
        // 获得对应的通用数组指针
        shared_ptr<universal_array> index_array_ptr = this->meta_data_set_ptr
                                                          ->get_element(pos_of_needed_index_array,
                                                                        real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array)
                                                          ->get_metadata_arr();

        // 将通用指针的数据类型挖掘出来
        data_type data_type_of_index = index_array_ptr->get_compress_data_type();

        // // 压缩数据类型，浮点类型不参与
        // if (get_config()["DATA_TYPE_COMPRESS"].as_bool() == true)
        // {
        //     cout << "code_generator::generate_matrix_format_read_code: data type compression is not supported" << endl;
        //     assert(false);
        // }

        if(data_type_of_index == FLOAT && get_config()["HALF"].as_bool() == true)
        {
            data_type_of_index = HALF;
        }

        // host指针的名字
        string host_format_index_ptr_name = get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array);
        string device_format_index_ptr_name = "d_" + host_format_index_ptr_name;
        // 数据的实际空间
        string format_index_array_size_expr = "sizeof(" + code_of_data_type(data_type_of_index) + ") * " + to_string(index_array_ptr->get_len());

        // 设备数组的声明
        // 首先是数据类型
        format_read_code << code_of_data_type(data_type_of_index) << "* ";
        // 然后是设备数组的指针
        format_read_code << device_format_index_ptr_name << ";" << endl;

        // 用这个指针来申请一个设备端数组
        format_read_code << "cudaMalloc(&" << device_format_index_ptr_name << ", " << format_index_array_size_expr << ");" << endl;
        // 拷贝主机端数组到设备端数组
        format_read_code << "cudaMemcpy(" << device_format_index_ptr_name << ", " << host_format_index_ptr_name << ", " << format_index_array_size_expr << ", "
                         << "cudaMemcpyHostToDevice"
                         << ");" << endl;
        format_read_code << endl;
    }

    format_read_code << endl;

    // 必然存在值数组
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->sub_matrix_id));
    // 得到将当前子矩阵的值数组
    shared_ptr<universal_array> val_arr_of_sub_matrix = this->meta_data_set_ptr
                                                            ->get_element(GLOBAL_META, "nz_vals", this->sub_matrix_id)
                                                            ->get_metadata_arr();

    // 整个矩阵的行数量
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "origin_row_num", -1));
    unsigned long row_num_of_the_whole_matrix = this->meta_data_set_ptr
                                                    ->get_element(GLOBAL_META, "origin_row_num", -1)
                                                    ->get_metadata_arr()
                                                    ->read_integer_from_arr(0);

    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "origin_col_num", -1));
    unsigned long col_num_of_the_whole_matrix = this->meta_data_set_ptr
                                                    ->get_element(GLOBAL_META, "origin_col_num", -1)
                                                    ->get_metadata_arr()
                                                    ->read_integer_from_arr(0);

    data_type data_type_of_vec = val_arr_of_sub_matrix->get_data_type();
    assert(data_type_of_vec == FLOAT || data_type_of_vec == DOUBLE);

    if(data_type_of_vec == FLOAT && get_config()["HALF"].as_bool() == true)
    {
        data_type_of_vec = HALF;
    }

    // y向量的大小，需要知道整个矩阵的行数量
    string size_of_y_vec_expr = "sizeof(" + code_of_data_type(data_type_of_vec) + ") * " + to_string(row_num_of_the_whole_matrix) + " * " + to_string(K);
    // x向量的大小，需要知道整个矩阵的列数量
    string size_of_x_vec_expr = "sizeof(" + code_of_data_type(data_type_of_vec) + ") * " + to_string(col_num_of_the_whole_matrix) + " * " + to_string(K);

    // 加入x与y两个向量，内容基本固定
    format_read_code << code_of_data_type(data_type_of_vec) << "* "
                     << "y_arr = (" << code_of_data_type(data_type_of_vec) << "*)malloc(" << size_of_y_vec_expr << ");" << endl;
    format_read_code << code_of_data_type(data_type_of_vec) << "* "
                     << "x_arr = (" << code_of_data_type(data_type_of_vec) << "*)malloc(" << size_of_x_vec_expr << ");" << endl;

    format_read_code << endl;

    // 两个向量的初始化
    format_read_code << "for (unsigned long i = 0; i < " << row_num_of_the_whole_matrix << " * " << K << "; i++)" << endl
                     << "{" << endl;
    format_read_code << "y_arr[i] = 0;" << endl
                     << "}" << endl;

    format_read_code << endl;

    format_read_code << "for (unsigned long i = 0; i < " << col_num_of_the_whole_matrix << " * " << K << "; i++)" << endl
                     << "{" << endl;
    format_read_code << "x_arr[i] = 1;" << endl
                     << "}" << endl;

    format_read_code << endl;

    // 声明x和y的设备指针
    format_read_code << code_of_data_type(data_type_of_vec) << "* "
                     << "d_y_arr;" << endl;
    format_read_code << code_of_data_type(data_type_of_vec) << "* "
                     << "d_x_arr;" << endl;

    format_read_code << endl;

    // 创建两个设备指针
    format_read_code << "cudaMalloc(&d_y_arr, " << size_of_y_vec_expr << ");" << endl;
    format_read_code << "cudaMalloc(&d_x_arr, " << size_of_x_vec_expr << ");" << endl;

    format_read_code << endl;

    // 拷贝
    format_read_code << "cudaMemcpy(d_y_arr, y_arr, " << size_of_y_vec_expr << ", cudaMemcpyHostToDevice);" << endl;
    format_read_code << "cudaMemcpy(d_x_arr, x_arr, " << size_of_x_vec_expr << ", cudaMemcpyHostToDevice);" << endl;

    return format_read_code.str();
}

string code_generator::generate_kernel_calling_code(unsigned int kernel_repeat_number)
{
    stringstream kernel_calling_code;

    assert(this->check());
    assert(kernel_repeat_number > 0);

    // 函数的调用，查看是不是要迭代
    if (kernel_repeat_number != 1)
    {
        kernel_calling_code << "for (int i = 0; i < " << kernel_repeat_number << "; i++)" << endl;

        kernel_calling_code << "{" << endl;
    }

    // 一般只有一个函数
    kernel_calling_code << "kernel_" << this->sub_matrix_id << "<<<grid_dim, block_dim>>>(";

    // 将所有的参数写入
    for (unsigned long i = 0; i < this->pos_of_needed_metadata_vec.size(); i++)
    {

        // 获取当前的设备指针
        POS_TYPE pos_of_needed_index_array = this->pos_of_needed_metadata_vec[i];
        string real_name_of_needed_index_array = this->real_name_of_needed_metadata_vec[i];
        int sub_matrix_id_of_needed_index_array = this->sub_matrix_id_of_needed_metadata_vec[i];

        // 当前依赖数据对应的设备数据指针
        string device_format_index_ptr_name = "d_" + get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array);

        // 调用一个指针
        kernel_calling_code << device_format_index_ptr_name;

        // 如果不是最后一个索引就加逗号
        // if (i < this->pos_of_needed_metadata_vec.size() - 1)
        // {
        kernel_calling_code << ", ";
        // }
    }

    // 再加上x和y两个数组
    kernel_calling_code << "d_y_arr, d_x_arr, K";

    kernel_calling_code << ");" << endl;

    if (kernel_repeat_number != 1)
    {
        kernel_calling_code << "}" << endl;
    }

    kernel_calling_code << "cudaDeviceSynchronize();" << endl;

    return kernel_calling_code.str();
}

string code_generator::generate_profiling_code_and_kernel(unsigned int kernel_repeat_number)
{
    stringstream profiling_kernel_calling_code;

    assert(this->check());
    unsigned long K = get_config()["DENSE_MATRIX_SIZE"].as_integer();

    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->sub_matrix_id));
    // 得到将当前子矩阵的值数组
    shared_ptr<universal_array> val_arr_of_sub_matrix = this->meta_data_set_ptr
                                                            ->get_element(GLOBAL_META, "nz_vals", this->sub_matrix_id)
                                                            ->get_metadata_arr();

    // 必然存在矩阵的原始大小的记录
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "origin_row_num", -1));
    data_type data_type_of_vec = val_arr_of_sub_matrix->get_data_type();
    unsigned long row_num_of_the_whole_matrix = this->meta_data_set_ptr
                                                    ->get_element(GLOBAL_META, "origin_row_num", -1)
                                                    ->get_metadata_arr()
                                                    ->read_integer_from_arr(0);


    assert(data_type_of_vec == FLOAT || data_type_of_vec == DOUBLE);

    if(data_type_of_vec == FLOAT && get_config()["HALF"].as_bool() == true)
    {
        data_type_of_vec = HALF;
    }
    profiling_kernel_calling_code << "dim3 grid_dim(" << this->thread_block_num[0] << ", " << this->thread_block_num[1] << ");" << endl;
    profiling_kernel_calling_code << "dim3 block_dim(" << this->thread_num[0] << ", " << this->thread_num[1] << ");" << endl;
    profiling_kernel_calling_code << "unsigned int K = "<< get_config()["DENSE_MATRIX_SIZE"].as_integer() << ";" << endl; 


    profiling_kernel_calling_code << this->generate_kernel_calling_code(1) << endl;
    // y向量的大小，需要知道整个矩阵的行数量
    string size_of_y_vec_expr = "sizeof(" + code_of_data_type(data_type_of_vec) + ") * " + to_string(row_num_of_the_whole_matrix) + " * " + to_string(K);

    // 拷贝输出
    profiling_kernel_calling_code << "cudaMemcpy(y_arr, d_y_arr, " << size_of_y_vec_expr << ", cudaMemcpyDeviceToHost);" << endl;

    // 处理计时同步开始
    profiling_kernel_calling_code << "struct timeval start,end;" << endl;
    profiling_kernel_calling_code << "cudaDeviceSynchronize();" << endl;
    profiling_kernel_calling_code << "gettimeofday(&start, NULL);" << endl
                                  << endl;

    // 处理内核函数调用
    profiling_kernel_calling_code << this->generate_kernel_calling_code(kernel_repeat_number) << endl;

    // 处理计时同步结束
    // profiling_kernel_calling_code << "cudaDeviceSynchronize();" << endl;
    profiling_kernel_calling_code << "gettimeofday(&end, NULL);" << endl
                                  << endl;

    // 必然存在值数组



    // profiling_kernel_calling_code << "unsigned long M, N, K, nnz;" << endl;

    // profiling_kernel_calling_code << get_config()["PRECISE_OF_FLOAT"].as_float() << "C_ref;" << endl;

    // profiling_kernel_calling_code << "spmm_reference_host<unsigne long," << get_config()["PRECISE_OF_FLOAT"].as_float() << ">(M, N, K, nnz, , , , B_h, C_ref);" << endl;

    // 获取当前子矩阵的非零元数量
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "origin_nnz_num", -1));

    unsigned long nnz_number_of_sub_matrix = this->meta_data_set_ptr
                                                 ->get_element(GLOBAL_META, "origin_nnz_num", -1)
                                                 ->get_metadata_arr()
                                                 ->read_integer_from_arr(0);

    // 计算性能
    profiling_kernel_calling_code << "long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;" << endl;
    profiling_kernel_calling_code << "double gflops = ((double)" << get_config()["FLOAT_RATE"].as_integer() << " * " << nnz_number_of_sub_matrix << " * " << K;
    profiling_kernel_calling_code << " * " << kernel_repeat_number << "/ ((double)timeuse / 1000000)) / 1000000000;" << endl;

    profiling_kernel_calling_code << endl;

    // 在命令行中打印出来性能
    profiling_kernel_calling_code << "cout << \"time = \" << timeuse /1000.0 << \" \" << \"gflops = \" << gflops << endl;" << endl;

    if (get_config()["HALF"].as_bool() == true)
    {
        profiling_kernel_calling_code << "int M;int KK;int nnz;vector<int> csr_indptr_buffer;vector<int> csr_indices_buffer;read_mtx_file(argv[1], M, KK, nnz, csr_indptr_buffer, csr_indices_buffer);int N = K;half *B_h = NULL, *csr_values_h = NULL, *C_ref = NULL;B_h = (half *)malloc(sizeof(half) * KK * N);C_ref = (half *)malloc(sizeof(half) * M * N);csr_values_h = (half *)malloc(sizeof(half) * nnz);fill_one_half(csr_values_h, nnz);fill_one_half(B_h, KK * N);spmm_reference_host<int, half>(M, N, KK, csr_indptr_buffer.data(),csr_indices_buffer.data(), csr_values_h, B_h,C_ref);bool correct = check_result<half>(M, N, y_arr, C_ref, true);" << endl;
    }
    else
    {
        profiling_kernel_calling_code << "int M;int KK;int nnz;vector<int> csr_indptr_buffer;vector<int> csr_indices_buffer;read_mtx_file(argv[1], M, KK, nnz, csr_indptr_buffer, csr_indices_buffer);int N = K;float *B_h = NULL, *csr_values_h = NULL, *C_ref = NULL;B_h = (float *)malloc(sizeof(float) * KK * N);C_ref = (float *)malloc(sizeof(float) * M * N);csr_values_h = (float *)malloc(sizeof(float) * nnz);fill_one(csr_values_h, nnz);fill_one(B_h, KK * N);spmm_reference_host<int, float>(M, N, KK, csr_indptr_buffer.data(),csr_indices_buffer.data(), csr_values_h, B_h,C_ref);bool correct = check_result<float>(M, N, y_arr, C_ref);" << endl;
    }

    profiling_kernel_calling_code << endl;

    // 将性能指标输出到文件中，文件所在的位置个代码在一个目录下，文件的目录
    string perf_file_name = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(this->output_id) + "/perf_result";

    // 创建一个输出流
    profiling_kernel_calling_code << "ofstream resultWrite(\"" << perf_file_name << "\", ios::out | ios::trunc);" << endl;
    profiling_kernel_calling_code << "resultWrite << timeuse /1000.0 << endl << gflops << endl;" << endl;
    profiling_kernel_calling_code << "resultWrite.close();" << endl;

    profiling_kernel_calling_code << endl;

    return profiling_kernel_calling_code.str();
}

string code_generator::generate_main_function_code(unsigned int kernel_repeat_number)
{
    stringstream main_fun_code;

    assert(this->check());

    main_fun_code << "int main(int argc, char** argv)" << endl
                  << "{" << endl;

    main_fun_code << this->generate_matrix_format_read_code() << endl;

    main_fun_code << this->generate_profiling_code_and_kernel(kernel_repeat_number) << endl;

    main_fun_code << "return 0;" << endl;

    main_fun_code << "}" << endl;

    return main_fun_code.str();
}

void code_generator::generate_kernel_file(unsigned int kernel_repeat_number)
{
    assert(this->check());
    // 首先判断之前是否已经输出了数据结构
    assert(this->output_id > 0);

    string dir_of_data_source = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(this->output_id);
    // 对应的目录确实存在
    assert(file_is_exist(dir_of_data_source));

    // 程序的头文件库必须存在
    string header_lib_path = get_config()["ROOT_PATH_STR"].as_string() + "/cuda_code/kernel_lib.hpp";
    // 编译脚本，拷贝到对应目录下cuda_code/make_kernel.sh
    string compile_script_path = get_config()["ROOT_PATH_STR"].as_string() + "/cuda_code/make_kernel.sh";
    assert(file_is_exist(header_lib_path));

    // 将头文件拷贝到对应的目录下
    system(("cp " + header_lib_path + " " + dir_of_data_source).c_str());
    // 将脚本拷贝到对应的目录下
    system(("cp " + compile_script_path + " " + dir_of_data_source).c_str());

    stringstream file_content;

    // 首先引入头文件
    file_content << "#include \"kernel_lib.hpp\"" << endl
                 << endl;

    file_content << endl
                 << this->generate_kernel_declaration_code() << endl;

    file_content << this->generate_main_function_code(kernel_repeat_number) << endl;

    // 用write_string_to_file写文件
    string kernel_file_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(this->output_id) + "/" + "kernel_file.cu";
    write_string_to_file(kernel_file_path, file_content.str());
}

unsigned long code_generator::write_matrix_format_to_disk()
{
    assert(this->check());
    // 之前没有输出
    assert(this->output_id == 0);

    // 将对应的数据写到disk中
    unsigned long output_id = this->meta_data_set_ptr->output_format_to_dir(this->pos_of_needed_metadata_vec, this->real_name_of_needed_metadata_vec, this->sub_matrix_id_of_needed_metadata_vec);
    assert(output_id != 0);

    this->output_id = output_id;

    return output_id;
}

void code_generator::set_thread_grid(vector<unsigned int> grid, vector<unsigned int> block)
{
    this->thread_block_num = grid;
    this->thread_num = block;
}

string code_generator::generate_header_of_kernel_declaration_code()
{
    assert(this->check());
    stringstream kernel_fun_header_code;

    kernel_fun_header_code << "__global__ void kernel_" << this->sub_matrix_id << "(";

    // 遍历所有需要的参数
    for (unsigned long i = 0; i < this->pos_of_needed_metadata_vec.size(); i++)
    {
        // 获取当前的设备指针
        POS_TYPE pos_of_needed_index_array = this->pos_of_needed_metadata_vec[i];
        string real_name_of_needed_index_array = this->real_name_of_needed_metadata_vec[i];
        int sub_matrix_id_of_needed_index_array = this->sub_matrix_id_of_needed_metadata_vec[i];

        // 获得当前通用指针
        assert(sub_matrix_id_of_needed_index_array == this->sub_matrix_id);

        // 当前数据在metadata set中必然存在
        assert(this->meta_data_set_ptr->is_exist(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array));
        // 获得对应的通用数组指针
        shared_ptr<universal_array> index_array_ptr = this->meta_data_set_ptr
                                                          ->get_element(pos_of_needed_index_array,
                                                                        real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array)
                                                          ->get_metadata_arr();

        // 指针的名字
        string format_index_ptr_name = get_metadata_item_name(pos_of_needed_index_array, real_name_of_needed_index_array, sub_matrix_id_of_needed_index_array);

        // 指针的数据类型
        data_type type_of_index_array = index_array_ptr->get_compress_data_type();

        if(type_of_index_array == FLOAT && get_config()["HALF"].as_bool() == true)
        {
            type_of_index_array = HALF;
        }

        kernel_fun_header_code << code_of_data_type(type_of_index_array) << "* " << format_index_ptr_name;

        // 如果不是最后一个，就加逗号
        // if (i < this->pos_of_needed_metadata_vec.size() - 1)
        // {
        kernel_fun_header_code << ", ";
        // }
    }

    // 增加x和y两个数组的指针
    // 必然存在值数组
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->sub_matrix_id));
    // 得到将当前子矩阵的值数组
    shared_ptr<universal_array> val_arr_of_sub_matrix = this->meta_data_set_ptr
                                                            ->get_element(GLOBAL_META, "nz_vals", this->sub_matrix_id)
                                                            ->get_metadata_arr();

    // x和y连个数组的数据类型
    data_type data_type_of_vec = val_arr_of_sub_matrix->get_data_type();
    assert(data_type_of_vec == FLOAT || data_type_of_vec == DOUBLE);

    if(data_type_of_vec == FLOAT && get_config()["HALF"].as_bool() == true)
    {
        data_type_of_vec = HALF;
    }
    kernel_fun_header_code << code_of_data_type(data_type_of_vec) << "* y_arr, ";

    kernel_fun_header_code << code_of_data_type(data_type_of_vec) << "* x_arr,";

    kernel_fun_header_code << code_of_data_type(UNSIGNED_INT) << " K";

    kernel_fun_header_code << ")";

    return kernel_fun_header_code.str();
}

string code_generator::generate_kernel_declaration_code()
{
    assert(this->check());
    stringstream kernel_fun_code;

    kernel_fun_code << this->generate_header_of_kernel_declaration_code() << endl
                    << "{" << endl;

    // 这里开始是kernel的实现
    kernel_fun_code << this->generate_code_of_grid_info_calculate() << endl;

    // 这里开始shared mem的声明
    kernel_fun_code << this->generate_code_of_shared_memory_array_declaration() << endl;

    // 这里声明全局变量
    kernel_fun_code << this->generate_global_var_init_code() << endl;

    // 查看这里有没有编译
    if (this->root_for_token_ptr == NULL || this->compiled_for_code == "")
    {
        cout << "code_generator::generate_kernel_declaration_code: not compiled" << endl;
    }

    // 如果已经编译了，那就打印
    kernel_fun_code << this->compiled_for_code;

    kernel_fun_code << endl
                    << "}" << endl;

    return kernel_fun_code.str();
}

string code_generator::generate_code_of_shared_memory_array_declaration()
{
    assert(this->check());
    stringstream shared_mem_declare_code;

    for (unsigned int i = 0; i < this->needed_shared_mem_data_type_vec.size(); i++)
    {
        // 声明一个内容
        shared_mem_declare_code << "__shared__ " << code_of_data_type(this->needed_shared_mem_data_type_vec[i]) << " " << this->needed_shared_mem_name_vec[i];
        shared_mem_declare_code << "[" << this->needed_shared_mem_size_vec[i] << "];" << endl;
    }

    return shared_mem_declare_code.str();
}

void code_generator::set_interleave_storage()
{
    this->is_interleave_storaged = true;
}

bool code_generator::get_interleave_storage()
{
    return this->is_interleave_storaged;
}

shared_ptr<var_name_token> code_generator::generate_global_var(data_type type, string var_name, shared_ptr<math_expr_token> var_init_expr_token, unsigned int size)
{
    if (size == 0)
    {
        assert(check_data_type(type) == true);

        if (var_init_expr_token != NULL)
        {
            assert(var_init_expr_token->static_check() == true);
        }

        // 首先查看当前变量没有出现过
        for (unsigned int i = 0; i < this->global_var_init_token_vec.size(); i++)
        {
            shared_ptr<basic_token> var_init_token_ptr = this->global_var_init_token_vec[i];
            string existing_var_name = var_init_token_ptr->get_inited_var_name();

            if (existing_var_name == var_name)
            {
                shared_ptr<var_name_token> var_name_token_ptr(new var_name_token(var_name, REGISTER_VAR_TYPE));
                return var_name_token_ptr;
            }
        }

        // 创建一个data type的token
        shared_ptr<data_type_token> data_type_token_ptr(new data_type_token(type, false));
        // 创建一个名字
        shared_ptr<var_name_token> var_name_token_ptr(new var_name_token(var_name, REGISTER_VAR_TYPE));

        // 创建一个init token
        shared_ptr<var_init_token> var_init_token_ptr(new var_init_token(data_type_token_ptr, var_name_token_ptr, var_init_expr_token));

        // 存到global_var_init_token_vec中
        this->global_var_init_token_vec.push_back(var_init_token_ptr);

        return var_name_token_ptr;
    }
    else
    {

        for (unsigned int i = 0; i < this->global_var_init_token_vec.size(); i++)
        {

            shared_ptr<basic_token> var_init_token_ptr = this->global_var_init_token_vec[i];
            string existing_var_name = var_init_token_ptr->get_inited_var_name();

            if (existing_var_name == var_name)
            {
                shared_ptr<var_name_token> var_name_token_ptr(new var_name_token(var_name, REGISTER_VAR_TYPE));
                return var_name_token_ptr;
            }

        }

        // 创建一个data type的token
        shared_ptr<data_type_token> data_type_token_ptr(new data_type_token(type, false));
        // 创建一个名字
        shared_ptr<var_name_token> var_name_token_ptr(new var_name_token(var_name, REGISTER_VAR_TYPE));
        shared_ptr<math_expr_token> arr_size(new math_expr_token(to_string(size)));

        // 创建一个init token
        shared_ptr<arr_declaration_token> arr_declaration_token_ptr(new arr_declaration_token(data_type_token_ptr, var_name_token_ptr, arr_size));
        // 存到global_var_init_token_vec中
        this->global_var_init_token_vec.push_back(arr_declaration_token_ptr);

        return var_name_token_ptr;
    }
}

bool code_generator::global_var_is_existing(string var_name)
{
    for (unsigned int i = 0; i < this->global_var_init_token_vec.size(); i++)
    {
        shared_ptr<basic_token> var_init_token_ptr = this->global_var_init_token_vec[i];
        assert(var_init_token_ptr->static_check() == true);

        string existing_var_name = var_init_token_ptr->get_inited_var_name();

        if (existing_var_name == var_name)
        {
            return true;
        }
    }

    return false;
}

string code_generator::total_thread_num_code()
{
    // 如果需要总线程数量，那么一定需要每个thread block中线程的数量和thread block的数量
    this->need_thread_number_in_thread_block = true;
    this->need_the_whole_thread_block_number = true;

    this->need_the_whole_thread_num = true;
    return "total_thd_num";
}

string code_generator::total_thread_block_num_code()
{
    // 可以直接查出对应的线程块数量，依赖于其他的数据
    this->need_the_whole_thread_block_number = true;

    return "total_tblk_num";
}

string code_generator::total_warp_num_code()
{
    // 直接根据总线程的数量来倒推warp的数量
    this->need_thread_number_in_thread_block = true;
    this->need_the_whole_thread_block_number = true;
    this->need_the_whole_thread_num = true;

    this->need_the_whole_warp_num = true;

    return "total_warp_num";
}

string code_generator::thread_num_in_thread_block_code()
{
    this->need_thread_number_in_thread_block = true;

    return "thd_num_in_thd_blk";
}

string code_generator::warp_num_in_thread_block_code()
{
    this->need_thread_number_in_thread_block = true;
    this->need_warp_number_in_thread_block = true;

    return "warp_num_in_thd_blk";
}

string code_generator::global_thread_block_id_code()
{
    this->need_global_thread_block_id = true;

    return "thd_blk_gid";
}

string code_generator::global_thread_id_code()
{
    this->need_global_thread_block_id = true;
    this->need_thread_number_in_thread_block = true;
    this->need_thread_id_in_thread_block = true;

    this->need_global_thread_id = true;

    return "thd_gid";
}

string code_generator::global_warp_id_code()
{
    this->need_global_thread_block_id = true;
    this->need_thread_number_in_thread_block = true;
    this->need_thread_id_in_thread_block = true;
    this->need_global_thread_id = true;

    this->need_global_warp_id = true;

    return "warp_gid";
}

string code_generator::warp_id_in_thread_block_code()
{
    this->need_thread_id_in_thread_block = true;
    this->need_warp_id_in_thread_block = true;

    return "warp_id_in_thd_blk";
}

string code_generator::thread_id_in_thread_block_code()
{
    this->need_thread_id_in_thread_block = true;

    return "thd_id_in_thd_blk";
}

string code_generator::thread_id_in_warp_code()
{
    this->need_thread_id_in_thread_block = true;
    this->need_thread_id_in_warp = true;

    return "thd_id_in_warp";
}

string code_generator::vector_width_code()
{
    return to_string(get_config()["VECTOR_WIDTH"].as_integer());
}

string code_generator::generate_code_of_grid_info_calculate()
{
    stringstream grid_info_code;

    // 线程网格信息的数据类型
    data_type info_type = UNSIGNED_INT;

    // 使用unsigned int的寄存器来记录所有的线程信息
    // 首先是线程块的数量和线程块内线程数量
    // 要注意代码生成的顺序
    if (this->need_the_whole_thread_block_number == true)
    {
        grid_info_code << code_of_data_type(info_type) << " " << this->total_thread_block_num_code();
        grid_info_code << " = gridDim.x * gridDim.y;" << endl;
    }

    if (this->need_thread_number_in_thread_block == true)
    {
        grid_info_code << code_of_data_type(info_type) << " " << this->thread_num_in_thread_block_code();
        grid_info_code << " = blockDim.x * blockDim.y;" << endl;
    }

    // 线程的总数量
    if (this->need_the_whole_thread_num == true)
    {
        assert(this->need_the_whole_thread_block_number == true);
        assert(this->need_thread_number_in_thread_block == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->total_thread_num_code();
        // 用线程块的数量和每个线程块线程的数量相乘就能得到
        grid_info_code << " = " << this->total_thread_block_num_code() << " * " << this->thread_num_in_thread_block_code() << ";" << endl;
    }

    // warp的总数量，用线程数量除32
    if (this->need_the_whole_warp_num == true)
    {
        assert(this->need_the_whole_thread_block_number == true);
        assert(this->need_thread_number_in_thread_block == true);
        assert(this->need_the_whole_thread_num == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->total_warp_num_code();
        // 线程数量除32
        grid_info_code << " = " << this->total_thread_num_code() << " / " << this->vector_width_code() << ";" << endl;
    }

    if (this->need_thread_id_in_thread_block == true)
    {
        grid_info_code << code_of_data_type(info_type) << " " << this->thread_id_in_thread_block_code();
        // 线程块内id可以直接获得
        grid_info_code << " = threadIdx.x + threadIdx.y * blockDim.x;" << endl;
    }

    // 一个线程块内部的warp id，用线程块内的线程的id除32
    if (this->need_warp_id_in_thread_block == true)
    {
        assert(this->need_thread_id_in_thread_block == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->warp_id_in_thread_block_code();
        // 用线程块内的id除向量的宽度
        grid_info_code << " = " << this->thread_id_in_thread_block_code() << " / " << this->vector_width_code() << ";" << endl;
    }

    // 线程块内部warp数量
    if (this->need_warp_number_in_thread_block == true)
    {
        assert(this->need_thread_number_in_thread_block == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->warp_num_in_thread_block_code();
        // 用线程块内线程的速度除32得到
        grid_info_code << " = " << this->thread_num_in_thread_block_code() << " / " << this->vector_width_code() << ";" << endl;
    }

    // 一个warp内部的thread id，处理的方式是用thread block内的thread id取32
    if (this->need_thread_id_in_warp == true)
    {
        assert(this->need_thread_id_in_thread_block == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->thread_id_in_warp_code();
        grid_info_code << " = " << this->thread_id_in_thread_block_code() << " % " << this->vector_width_code() << ";" << endl;
    }

    // 线程块的全局id
    if (this->need_global_thread_block_id == true)
    {
        grid_info_code << code_of_data_type(info_type) << " " << this->global_thread_block_id_code();
        // 直接获得
        grid_info_code << " = blockIdx.x + blockIdx.y * gridDim.x;" << endl;
    }

    // 线程的全局id
    if (this->need_global_thread_id == true)
    {
        assert(this->need_thread_number_in_thread_block == true);
        assert(this->need_global_thread_block_id == true);
        assert(this->need_thread_id_in_thread_block == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->global_thread_id_code();
        // 用每个thread block的线程数量，当前的thread block id和thread block内的thread id来计算thread的全局id
        grid_info_code << " = " << this->global_thread_block_id_code() << " * " << this->thread_num_in_thread_block_code();
        grid_info_code << " + " << this->thread_id_in_thread_block_code() << ";" << endl;
    }

    // warp的全局id
    if (this->need_global_warp_id == true)
    {
        assert(this->need_thread_number_in_thread_block == true);
        assert(this->need_global_thread_block_id == true);
        assert(this->need_thread_id_in_thread_block == true);
        assert(this->need_global_thread_id == true);

        grid_info_code << code_of_data_type(info_type) << " " << this->global_warp_id_code();
        grid_info_code << " = " << this->global_thread_id_code() << " / " << this->vector_width_code() << ";" << endl;
    }

    return grid_info_code.str();
}

string code_generator::generate_global_var_init_code()
{
    stringstream init_code;

    // 遍历所有的全局变量
    for (unsigned int i = 0; i < this->global_var_init_token_vec.size(); i++)
    {
        init_code << this->global_var_init_token_vec[i]->run() << endl;
    }

    return init_code.str();
}

shared_ptr<var_name_token> code_generator::code_of_data_block_id_distributed_to_spec_paral(POS_TYPE pos)
{
    assert(check_pos_type(pos) == true);

    if (pos == TBLOCK_META)
    {
        return this->BMTB_id;
    }

    if (pos == WARP_META)
    {
        return this->BMW_id;
    }

    if (pos == THREAD_META)
    {
        return this->BMT_id;
    }

    cout << "code_generator::code_of_data_block_id_distributed_to_spec_paral: pos is illegal, pos:" << convert_pos_type_to_string(pos) << endl;
    assert(false);
    return NULL;
}

shared_ptr<metadata_set_get_token> code_generator::generate_token_of_fused_metadata_get_in_spec_paral_level(POS_TYPE pos)
{
    assert(check_pos_type(pos) == true);
    assert(this->check() == true);

    shared_ptr<metadata_set_get_token> return_token(new metadata_set_get_token(pos));

    // 如果是TBLOCK级别的并且需要使用shared memory广播的优化
    // if (pos == TBLOCK_META && get_config()["SHARED_MEM_BROADCASR_OPTIMIZATION"].as_bool() == true)
    // {
    //     cout << "code_generator::generate_token_of_metadata_get_in_spec_paral_level: not support SHARED_MEM_BROADCASR_OPTIMIZATION" << endl;
    //     assert(false);
    // }
    // else
    // {
    // 遍历当前并行层次的全部的的metadata索引，获得对应代码。
    for (int i = 0; i < this->pos_of_fused_metadata_vec.size(); i++)
    {
        // 查看当前的位置是不是正确
        if (this->pos_of_fused_metadata_vec[i] == pos)
        {
            // 获得名字和访问的表达式
            string real_name = this->real_name_of_fused_metadata_vec[i];
            shared_ptr<math_expr_token> access_index_expr = this->access_index_fused_metadata_vec[i];
            assert(access_index_expr->static_check() == true);

            // 这里对应的内容需要被输出，首先先输出对应变量的初始化
            // 获得变量名
            string var_of_readed_content = var_of_metadata_from_spec_paral(pos, real_name, this->sub_matrix_id, access_index_expr);

            // 对应的内容在
            assert(this->meta_data_set_ptr->is_exist(pos, real_name, this->sub_matrix_id) == true);

            // 将要读的数据类型读出，查看数据类型
            shared_ptr<universal_array> index_array_ptr = this->meta_data_set_ptr
                                                              ->get_element(pos, real_name, this->sub_matrix_id)
                                                              ->get_metadata_arr();

            // 查看数据类型
            data_type data_type_of_index = index_array_ptr->get_compress_data_type();

            // // 查看是否要压缩
            // if (get_config()["DATA_TYPE_COMPRESS"].as_bool() == true)
            // {
            //     cout << "code_generator::generate_token_of_funsed_metadata_get_in_spec_paral_level: data type compression has not supported" << endl;
            //     assert(false);
            // }

            // 声明一个数据类型的token
            shared_ptr<data_type_token> type_of_readed_data(new data_type_token(data_type_of_index, false));

            // 一个变量名的token
            shared_ptr<var_name_token> var_token_of_readed_content(new var_name_token(var_of_readed_content, REGISTER_VAR_TYPE));

            // 一个初始化的token
            shared_ptr<var_init_token> token_of_var_init(new var_init_token(type_of_readed_data, var_token_of_readed_content, NULL));

            // // 创建一个数组名
            // string mem_access_arr_name = get_metadata_item_name(pos, real_name, this->sub_matrix_id);
            // // 创建一个数组名的token
            // shared_ptr<var_name_token> arr_var_name_token(new var_name_token(mem_access_arr_name, GLOBAL_MEM_VAR_TYPE));

            // shared_ptr<arr_access_token> metadata_access_token(new arr_access_token(var_token_of_readed_content, arr_var_name_token, access_index_expr));
            shared_ptr<basic_token> metadata_access_token = get_compress_and_relative_result(pos, real_name, this->sub_matrix_id, var_token_of_readed_content, access_index_expr);

            assert(token_of_var_init->static_check() == true);
            assert(metadata_access_token->static_check() == true);

            // 将对应的两行代码放到metadata get token中
            return_token->add_metadata_get_expr(token_of_var_init);

            return_token->add_metadata_get_expr(metadata_access_token);
        }
    }
    // }

    return return_token;
}

shared_ptr<var_name_token> code_generator::generate_fused_memory_access_with_relative(POS_TYPE pos, string metadata_name, shared_ptr<math_expr_token> mem_access_index_expr)
{

    string real_metadata_name_1 = metadata_name + "_relative_to_BMW";
    string real_metadata_name_2 = metadata_name + "_relative_to_BMTB";
    if(metadata_name.find("without") != string::npos)
    {
        real_metadata_name_1 = "first_row_indices_relative_to_BMW";
        real_metadata_name_2 = "first_row_indices_relative_to_BMTB";
    }
    
    POS_TYPE parent_pos;
    bool end_loop = false;
    if (mem_access_index_expr->run().find("+1") != string::npos || mem_access_index_expr->run().find("+ 1") != string::npos)
    {
        end_loop = true;
    }

    if (this->meta_data_set_ptr->is_exist(pos, real_metadata_name_1, this->sub_matrix_id) == true)
    {
        parent_pos = WARP_META;
        shared_ptr<math_expr_token> parent_id_token;
        if (end_loop == true)
        {
            parent_id_token = make_shared<math_expr_token>(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run() + "+1");
        }
        else
        {
            parent_id_token = make_shared<math_expr_token>(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run());
        }

        shared_ptr<var_name_token> parent_result = generate_fused_memory_access_with_relative(parent_pos, metadata_name, parent_id_token);
        shared_ptr<var_name_token> relative_result = generate_fused_memory_access(pos, real_metadata_name_1, mem_access_index_expr);
        string var_of_readed_content = var_of_metadata_from_spec_paral(pos, metadata_name, this->sub_matrix_id, mem_access_index_expr);
        shared_ptr<var_name_token> return_var_name_token(new var_name_token(var_of_readed_content, REGISTER_VAR_TYPE));
        this->add_new_fused_metadata_access(pos, metadata_name, this->sub_matrix_id, mem_access_index_expr);
        return return_var_name_token;
    }
    else if (this->meta_data_set_ptr->is_exist(pos, real_metadata_name_2, this->sub_matrix_id) == true)
    {
        parent_pos = TBLOCK_META;
        shared_ptr<math_expr_token> parent_id_token;
        if (end_loop == true)
        {
            parent_id_token = make_shared<math_expr_token>(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run() + "+1");
        }
        else
        {
            parent_id_token = make_shared<math_expr_token>(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run());
        }

        shared_ptr<var_name_token> parent_result = generate_fused_memory_access_with_relative(parent_pos, metadata_name, parent_id_token);
        shared_ptr<var_name_token> relative_result = generate_fused_memory_access(pos, real_metadata_name_2, mem_access_index_expr);
        string var_of_readed_content = var_of_metadata_from_spec_paral(pos, metadata_name, this->sub_matrix_id, mem_access_index_expr);
        shared_ptr<var_name_token> return_var_name_token(new var_name_token(var_of_readed_content, REGISTER_VAR_TYPE));
        this->add_new_fused_metadata_access(pos, metadata_name, this->sub_matrix_id, mem_access_index_expr);
        return return_var_name_token;
    }
    else
    {
        return generate_fused_memory_access(pos, metadata_name, mem_access_index_expr);
    }
}

shared_ptr<var_name_token> code_generator::generate_fused_memory_access(POS_TYPE pos, string metadata_name, shared_ptr<math_expr_token> mem_access_index_expr)
{
    // 增加一个合并之后的访问
    assert(check_pos_type(pos) == true);
    assert(sub_matrix_id >= 0);
    assert(this->meta_data_set_ptr->check() == true);
    assert(this->check() == true);
    // assert(this->sub_matrix_id == sub_matrix_id);
    // 输入的表达式合法
    assert(mem_access_index_expr != NULL);
    assert(mem_access_index_expr->static_check() == true);

    if (this->meta_data_set_ptr->is_exist(pos, metadata_name, this->sub_matrix_id) == false)
    {
        metadata_name = metadata_name + "_without_ending";
    }
    
    // cout << pos << " " << metadata_name << " " << this->sub_matrix_id << endl;
    assert(this->meta_data_set_ptr->is_exist(pos, metadata_name, this->sub_matrix_id) == true);

    // 通过读取的索引和位置生成对应的变量名
    string var_of_readed_content = var_of_metadata_from_spec_paral(pos, metadata_name, this->sub_matrix_id, mem_access_index_expr);

    // 使用这个名字初始化一个变量名，是一个寄存器变量
    shared_ptr<var_name_token> return_var_name_token(new var_name_token(var_of_readed_content, REGISTER_VAR_TYPE));
    assert(return_var_name_token->static_check() == true);


    // 如果其没有被记录到被需要的元数据中
    if (this->if_dependency_exist(pos, metadata_name, sub_matrix_id) == false)
    {
        // 添加一个被需要的元数据记录
        this->add_new_metadata_dependency(pos, metadata_name, sub_matrix_id);
    }

    // 记录需要合并的访存
    this->add_new_fused_metadata_access(pos, metadata_name, sub_matrix_id, mem_access_index_expr);

    return return_var_name_token;
}

vector<shared_ptr<basic_token>> code_generator::generate_unfused_memory_access(POS_TYPE pos, string metadata_name, shared_ptr<math_expr_token> mem_access_index_expr, bool flag, string row_name)
{
    assert(check_pos_type(pos) == true);
    assert(sub_matrix_id >= 0);
    assert(this->meta_data_set_ptr->check() == true);
    assert(this->check() == true);
    assert(mem_access_index_expr != NULL);
    assert(mem_access_index_expr->static_check() == true);

    if (this->meta_data_set_ptr->is_exist(pos, metadata_name, this->sub_matrix_id) == false)
    {
        metadata_name = metadata_name + "_without_ending";
    }

    assert(this->meta_data_set_ptr->is_exist(pos, metadata_name, this->sub_matrix_id) == true);

    vector<shared_ptr<basic_token>> return_token_vec;
    shared_ptr<universal_array> index_array_ptr = this->meta_data_set_ptr
                                                      ->get_element(pos, metadata_name, this->sub_matrix_id)
                                                      ->get_metadata_arr();

    if (this->if_dependency_exist(pos, metadata_name, this->sub_matrix_id) == false)
    {
        this->add_new_metadata_dependency(pos, metadata_name, this->sub_matrix_id);
    }

    data_type type_of_index_data = index_array_ptr->get_compress_data_type();

    if(type_of_index_data == FLOAT && get_config()["HALF"].as_bool() == true)
    {
        type_of_index_data = HALF;
    }

    string var_of_readed_content = var_of_metadata_from_spec_paral(pos, metadata_name, this->sub_matrix_id, mem_access_index_expr) + "_unfuse";
    shared_ptr<var_name_token> readed_var_name_token(new var_name_token(var_of_readed_content, REGISTER_VAR_TYPE));
    shared_ptr<data_type_token> data_type_of_readed_data(new data_type_token(type_of_index_data, false));
    shared_ptr<var_init_token> var_init_of_readed_data(new var_init_token(data_type_of_readed_data, readed_var_name_token, NULL));
    shared_ptr<basic_token> mem_access_token = get_compress_and_relative_result(pos, metadata_name, this->sub_matrix_id, readed_var_name_token, mem_access_index_expr);

    if (flag == false || metadata_name.find("row") == string::npos)
    {
        return_token_vec.push_back(var_init_of_readed_data);
        return_token_vec.push_back(mem_access_token);
    }
    else if (flag == true && metadata_name.find("row") != string::npos)
    {
        vector<shared_ptr<basic_token>> init_vec;
        vector<shared_ptr<basic_token>> assign_vec;
        shared_ptr<data_type_token> data_type_of_recovery(new data_type_token(UNSIGNED_LONG, false));
        if (row_name == "")
        {
            shared_ptr<var_init_token> var_init_of_recovery(new var_init_token(data_type_of_recovery, readed_var_name_token, NULL));
            init_vec.push_back(var_init_of_recovery);
            assign_vec.push_back(mem_access_token);
            if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", this->sub_matrix_id) == true)
            {
                this->add_new_metadata_dependency(GLOBAL_META, "original_nz_row_indices", this->sub_matrix_id);
                shared_ptr<math_expr_token> origin_index_math(new math_expr_token(var_of_readed_content));
                shared_ptr<basic_token> mem_access_token_origin = get_compress_and_relative_result(GLOBAL_META, "original_nz_row_indices", this->sub_matrix_id, readed_var_name_token, origin_index_math);
                assign_vec.push_back(mem_access_token_origin);
            }

            unsigned long boundary = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
            string boundary_string = readed_var_name_token->run() + " + " + to_string(boundary);
            shared_ptr<math_expr_token> boundary_recovery_math(new math_expr_token(boundary_string));
            shared_ptr<var_assign_token> div_recovery_assign(new var_assign_token(readed_var_name_token, boundary_recovery_math));
            if (boundary != 0)
            {
                assign_vec.push_back(div_recovery_assign);
            }

            if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", 0) == true && this->sub_matrix_id != 0)
            {
                this->add_new_metadata_dependency(GLOBAL_META, "original_nz_row_indices", 0);
                shared_ptr<math_expr_token> origin_index_math(new math_expr_token(var_of_readed_content));
                shared_ptr<basic_token> mem_access_token_origin = get_compress_and_relative_result(GLOBAL_META, "original_nz_row_indices", 0, readed_var_name_token, origin_index_math);
                assign_vec.push_back(mem_access_token_origin);
            }

            for (int i = 0; i < init_vec.size(); i++)
            {
                return_token_vec.push_back(init_vec[i]);
            }

            for (int j = 0; j < assign_vec.size(); j++)
            {
                return_token_vec.push_back(assign_vec[j]);
            }
        }
        else
        {
            shared_ptr<var_name_token> _readed_var_name_token(new var_name_token("origin_" + row_name, REGISTER_VAR_TYPE));
            shared_ptr<math_expr_token> init_name(new math_expr_token(row_name));
            shared_ptr<var_init_token> var_init_of_recovery(new var_init_token(data_type_of_recovery, _readed_var_name_token, init_name));
            init_vec.push_back(var_init_of_recovery);
            if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", this->sub_matrix_id) == true)
            {
                this->add_new_metadata_dependency(GLOBAL_META, "original_nz_row_indices", this->sub_matrix_id);
                shared_ptr<math_expr_token> origin_index_math(new math_expr_token(row_name));
                shared_ptr<basic_token> mem_access_token_origin = get_compress_and_relative_result(GLOBAL_META, "original_nz_row_indices", this->sub_matrix_id, _readed_var_name_token, origin_index_math);
                assign_vec.push_back(mem_access_token_origin);
            }

            unsigned long boundary = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
            string boundary_string = _readed_var_name_token->run() + " + " + to_string(boundary);
            shared_ptr<math_expr_token> boundary_recovery_math(new math_expr_token(boundary_string));
            shared_ptr<var_assign_token> div_recovery_assign(new var_assign_token(_readed_var_name_token, boundary_recovery_math));
            if (boundary != 0)
            {
                assign_vec.push_back(div_recovery_assign);
            }

            if (this->meta_data_set_ptr->is_exist(GLOBAL_META, "original_nz_row_indices", 0) == true && this->sub_matrix_id != 0)
            {
                this->add_new_metadata_dependency(GLOBAL_META, "original_nz_row_indices", 0);
                shared_ptr<math_expr_token> origin_index_math(new math_expr_token(var_of_readed_content));
                shared_ptr<basic_token> mem_access_token_origin = get_compress_and_relative_result(GLOBAL_META, "original_nz_row_indices", 0, _readed_var_name_token, origin_index_math);
                assign_vec.push_back(mem_access_token_origin);
            }

            for (int i = 0; i < init_vec.size(); i++)
            {
                return_token_vec.push_back(init_vec[i]);
            }

            for (int j = 0; j < assign_vec.size(); j++)
            {
                return_token_vec.push_back(assign_vec[j]);
            }
        }
    }

    return return_token_vec;
}

void code_generator::generate_loop_var(bool thread_x_direction_reuse)
{
    if (thread_x_direction_reuse == true)
    {
        if (this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == true)
        {
            shared_ptr<math_expr_token> tblock_begin_init(new math_expr_token("blockIdx.x"));
            shared_ptr<math_expr_token> tblock_step_init(new math_expr_token("gridDim.x"));
            shared_ptr<var_name_token> tblock_begin = this->generate_global_var(UNSIGNED_INT, "block_begin_ptr", tblock_begin_init);
            shared_ptr<var_name_token> tblcok_step = this->generate_global_var(UNSIGNED_INT, "block_step", tblock_step_init);
            this->set_for_loop_begin_ptr(TBLOCK_META, tblock_begin);
            this->set_for_loop_step(TBLOCK_META, tblcok_step);
        }

        if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == true)
        {
            if (this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == true)
            {
                shared_ptr<math_expr_token> warp_begin_init(new math_expr_token("threadIdx.y / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<math_expr_token> warp_step_init(new math_expr_token("blockDim.y / "+ get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<var_name_token> warp_begin = this->generate_global_var(UNSIGNED_INT, "warp_begin_ptr", warp_begin_init);
                shared_ptr<var_name_token> warp_step = this->generate_global_var(UNSIGNED_INT, "warp_step", warp_step_init);
                this->set_for_loop_begin_ptr(WARP_META, warp_begin);
                this->set_for_loop_step(WARP_META, warp_step);
            }
            else
            {
                shared_ptr<math_expr_token> warp_begin_init(new math_expr_token("blockIdx.x * blockDim.y / " + get_config()["VECTOR_WIDTH"].as_string() + " + threadIdx.y / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<math_expr_token> warp_step_init(new math_expr_token("gridDim.x *  blockDim.y / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<var_name_token> warp_begin = this->generate_global_var(UNSIGNED_INT, "warp_begin_ptr", warp_begin_init);
                shared_ptr<var_name_token> warp_step = this->generate_global_var(UNSIGNED_INT, "warp_step", warp_step_init);
                this->set_for_loop_begin_ptr(WARP_META, warp_begin);
                this->set_for_loop_step(WARP_META, warp_step);
            }
        }

        if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == false && this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == false)
        {
            shared_ptr<math_expr_token> thread_begin_init(new math_expr_token("(blockIdx.x * blockDim.y) + threadIdx.y"));
            shared_ptr<math_expr_token> thread_step_init(new math_expr_token("gridDim.x * blockDim.y"));
            shared_ptr<var_name_token> thread_begin = this->generate_global_var(UNSIGNED_INT, "thread_begin_ptr", thread_begin_init);
            shared_ptr<var_name_token> thread_step = this->generate_global_var(UNSIGNED_INT, "thread_step", thread_step_init);
            this->set_for_loop_begin_ptr(THREAD_META, thread_begin);
            this->set_for_loop_step(THREAD_META, thread_step);
        }
        else if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == true)
        {
            shared_ptr<math_expr_token> thread_begin_init(new math_expr_token("threadIdx.y % " + get_config()["VECTOR_WIDTH"].as_string()));
            shared_ptr<math_expr_token> thread_step_init(new math_expr_token(get_config()["VECTOR_WIDTH"].as_string()));
            shared_ptr<var_name_token> thread_begin = this->generate_global_var(UNSIGNED_INT, "thread_begin_ptr", thread_begin_init);
            shared_ptr<var_name_token> thread_step = this->generate_global_var(UNSIGNED_INT, "thread_step", thread_step_init);
            this->set_for_loop_begin_ptr(THREAD_META, thread_begin);
            this->set_for_loop_step(THREAD_META, thread_step);
        }
        else if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == false && this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == true)
        {
            shared_ptr<math_expr_token> thread_begin_init(new math_expr_token("threadIdx.y"));
            shared_ptr<math_expr_token> thread_step_init(new math_expr_token("blockDim.y"));
            shared_ptr<var_name_token> thread_begin = this->generate_global_var(UNSIGNED_INT, "thread_begin_ptr", thread_begin_init);
            shared_ptr<var_name_token> thread_step = this->generate_global_var(UNSIGNED_INT, "thread_step", thread_step_init);
            this->set_for_loop_begin_ptr(THREAD_META, thread_begin);
            this->set_for_loop_step(THREAD_META, thread_step);
        }
    }
    else
    {
        if (this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == true)
        {
            shared_ptr<math_expr_token> tblock_begin_init(new math_expr_token("blockIdx.x"));
            shared_ptr<math_expr_token> tblock_step_init(new math_expr_token("gridDim.x"));
            shared_ptr<var_name_token> tblock_begin = this->generate_global_var(UNSIGNED_INT, "block_begin_ptr", tblock_begin_init);
            shared_ptr<var_name_token> tblcok_step = this->generate_global_var(UNSIGNED_INT, "block_step", tblock_step_init);
            this->set_for_loop_begin_ptr(TBLOCK_META, tblock_begin);
            this->set_for_loop_step(TBLOCK_META, tblcok_step);
        }

        if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == true)
        {
            if (this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == true)
            {
                shared_ptr<math_expr_token> warp_begin_init(new math_expr_token("threadIdx.x / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<math_expr_token> warp_step_init(new math_expr_token("blockDim.x / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<var_name_token> warp_begin = this->generate_global_var(UNSIGNED_INT, "warp_begin_ptr", warp_begin_init);
                shared_ptr<var_name_token> warp_step = this->generate_global_var(UNSIGNED_INT, "warp_step", warp_step_init);
                this->set_for_loop_begin_ptr(WARP_META, warp_begin);
                this->set_for_loop_step(WARP_META, warp_step);
            }
            else
            {
                shared_ptr<math_expr_token> warp_begin_init(new math_expr_token("blockIdx.x * blockDim.x / " + get_config()["VECTOR_WIDTH"].as_string() + " + threadIdx.x / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<math_expr_token> warp_step_init(new math_expr_token("gridDim.x *  blockDim.x / " + get_config()["VECTOR_WIDTH"].as_string()));
                shared_ptr<var_name_token> warp_begin = this->generate_global_var(UNSIGNED_INT, "warp_begin_ptr", warp_begin_init);
                shared_ptr<var_name_token> warp_step = this->generate_global_var(UNSIGNED_INT, "warp_step", warp_step_init);
                this->set_for_loop_begin_ptr(WARP_META, warp_begin);
                this->set_for_loop_step(WARP_META, warp_step);
            }
        }

        if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == false && this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == false)
        {
            shared_ptr<math_expr_token> thread_begin_init(new math_expr_token("(blockIdx.x * blockDim.x) + threadIdx.x"));
            shared_ptr<math_expr_token> thread_step_init(new math_expr_token("gridDim.x * blockDim.x"));
            shared_ptr<var_name_token> thread_begin = this->generate_global_var(UNSIGNED_INT, "thread_begin_ptr", thread_begin_init);
            shared_ptr<var_name_token> thread_step = this->generate_global_var(UNSIGNED_INT, "thread_step", thread_step_init);
            this->set_for_loop_begin_ptr(THREAD_META, thread_begin);
            this->set_for_loop_step(THREAD_META, thread_step);
        }
        else if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == true)
        {
            shared_ptr<math_expr_token> thread_begin_init(new math_expr_token("threadIdx.x % " + get_config()["VECTOR_WIDTH"].as_string()));
            shared_ptr<math_expr_token> thread_step_init(new math_expr_token(get_config()["VECTOR_WIDTH"].as_string()));
            shared_ptr<var_name_token> thread_begin = this->generate_global_var(UNSIGNED_INT, "thread_begin_ptr", thread_begin_init);
            shared_ptr<var_name_token> thread_step = this->generate_global_var(UNSIGNED_INT, "thread_step", thread_step_init);
            this->set_for_loop_begin_ptr(THREAD_META, thread_begin);
            this->set_for_loop_step(THREAD_META, thread_step);
        }
        else if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id) == false && this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id) == true)
        {
            shared_ptr<math_expr_token> thread_begin_init(new math_expr_token("threadIdx.x"));
            shared_ptr<math_expr_token> thread_step_init(new math_expr_token("blockDim.x"));
            shared_ptr<var_name_token> thread_begin = this->generate_global_var(UNSIGNED_INT, "thread_begin_ptr", thread_begin_init);
            shared_ptr<var_name_token> thread_step = this->generate_global_var(UNSIGNED_INT, "thread_step", thread_step_init);
            this->set_for_loop_begin_ptr(THREAD_META, thread_begin);
            this->set_for_loop_step(THREAD_META, thread_step);
        }
    }
}

shared_ptr<for_token> code_generator::generate_for_token_of_spec_paral_level(POS_TYPE pos, shared_ptr<for_token> child_for_token, bool the_outest_loop)
{
    assert(check_pos_type(pos) == true);
    assert(this->check() == true);

    // 根据并行级别的不同创建需要使用不同的循环哨兵变量
    shared_ptr<var_name_token> iter_var_name = NULL;

    // 获取父并行层次
    POS_TYPE parent_pos = NONE_META;

    // 只有三个层次的并行，确定对应的哨兵变量
    if (pos == TBLOCK_META)
    {
        assert(the_outest_loop == true);
        if (child_for_token != NULL)
        {
            assert(child_for_token->get_token_position() == THREAD_META || child_for_token->get_token_position() == WARP_META);
        }

        assert(this->need_tblock_level_paral == true);

        iter_var_name = this->code_of_data_block_id_distributed_to_spec_paral(TBLOCK_META);
        parent_pos = GLOBAL_META;
    }
    else if (pos == WARP_META)
    {
        if (child_for_token != NULL)
        {
            assert(child_for_token->get_token_position() == THREAD_META);
        }

        iter_var_name = this->code_of_data_block_id_distributed_to_spec_paral(WARP_META);

        if (this->need_tblock_level_paral == true)
        {
            parent_pos = TBLOCK_META;
        }
        else
        {
            assert(the_outest_loop == true);
            parent_pos = GLOBAL_META;
        }

        assert(this->need_warp_level_paral == true);
    }
    else if (pos == THREAD_META)
    {
        assert(child_for_token == NULL);

        iter_var_name = this->code_of_data_block_id_distributed_to_spec_paral(THREAD_META);

        // 根据对并行级别的需要来决定
        if (this->need_warp_level_paral == true)
        {
            parent_pos = WARP_META;
        }
        else if (this->need_tblock_level_paral == true)
        {
            parent_pos = TBLOCK_META;
        }
        else
        {
            assert(the_outest_loop == true);
            parent_pos = GLOBAL_META;
        }

        assert(this->need_thread_level_paral == true);
    }
    else
    {
        cout << "code_generator::generate_for_token_of_spec_paral_level: the paral level is not support, pos:"
             << convert_pos_type_to_string(pos) << endl;
        assert(false);
    }

    if (the_outest_loop == true)
    {
        parent_pos = GLOBAL_META;
    }

    // 首先是哨兵变量的类型
    shared_ptr<data_type_token> data_type_of_iter_var_token(new data_type_token(UNSIGNED_INT, false));

    bool thread_x_reuse = (this->warp_level_reduction_token_ptr == NULL);
    this->generate_loop_var(thread_x_reuse);

    shared_ptr<basic_token> begin_loop_var_name = NULL;

    if (the_outest_loop == true)
    {

        if (pos == TBLOCK_META)
        {
            begin_loop_var_name = this->BMTB_begin_ptr;
        }
        else if (pos == WARP_META)
        {
            begin_loop_var_name = this->BMW_begin_ptr;
        }
        else if (pos == THREAD_META)
        {
            begin_loop_var_name = this->BMT_begin_ptr;
        }
    }
    else
    {
        assert(parent_pos != GLOBAL_META && parent_pos != NONE_META);
        // 父数据块的索引
        shared_ptr<var_name_token> parent_data_blk_id_token = this->code_of_data_block_id_distributed_to_spec_paral(parent_pos);

        shared_ptr<math_expr_token> parent_data_blk_id_expr_token(new math_expr_token(parent_data_blk_id_token->run()));

        // 当前并行级别的数据块的起始位置所在的metadata索引名
        string indices_name_of_blk_offset = "";
        if (pos == THREAD_META)
        {
            indices_name_of_blk_offset = "first_BMT_indices";
        }
        else if (pos == WARP_META)
        {
            indices_name_of_blk_offset = "first_BMW_indices";
        }
        else
        {
            assert(false);
        }

        // 使用这个索引来获得当前块在父块中的偏移量，这个偏移量用作循环的起始位置
        shared_ptr<var_name_token> begin_loop_var_name_part = this->generate_fused_memory_access(parent_pos, indices_name_of_blk_offset, parent_data_blk_id_expr_token);

        if (pos == TBLOCK_META)
        {
            shared_ptr<math_expr_token> begin_loop_var_name_expr(new math_expr_token(begin_loop_var_name_part->run() + " + " + this->BMTB_begin_ptr->run()));
            begin_loop_var_name = begin_loop_var_name_expr;
        }
        else if (pos == WARP_META)
        {
            shared_ptr<math_expr_token> begin_loop_var_name_expr(new math_expr_token(begin_loop_var_name_part->run() + " + " + this->BMW_begin_ptr->run()));
            begin_loop_var_name = begin_loop_var_name_expr;
        }
        else if (pos == THREAD_META)
        {
            shared_ptr<math_expr_token> begin_loop_var_name_expr(new math_expr_token(begin_loop_var_name_part->run() + " + " + this->BMT_begin_ptr->run()));
            begin_loop_var_name = begin_loop_var_name_expr;
        }
    }

    // 获得结束位置
    shared_ptr<var_name_token> end_loop_var_name = NULL;

    // 如果是最外层，那么结束位置就是当前并行级别所对应的数据块的数量
    if (the_outest_loop == true)
    {
        // 获取当前并行粒度的块数量，使用first_nz_indices来判断数据块数量
        // 首先查看首个非零元索引是不是存在的
        assert(this->meta_data_set_ptr->is_exist(pos, "first_nz_indices", this->sub_matrix_id));
        // 获得对应的数组的指针
        shared_ptr<universal_array> first_nz_indices = this->meta_data_set_ptr->get_element(pos, "first_nz_indices", this->sub_matrix_id)->get_metadata_arr();
        // 推导出这个级别数据块的个数
        unsigned int block_num = first_nz_indices->get_len() - 1;
        end_loop_var_name = make_shared<var_name_token>(to_string(block_num), CONSTANT_VAR_TYPE);
    }
    else
    {
        assert(parent_pos != GLOBAL_META && parent_pos != NONE_META);
        // 父块的数据块索引变量
        shared_ptr<var_name_token> parent_data_blk_id_token = this->code_of_data_block_id_distributed_to_spec_paral(parent_pos);
        // 用索引变量加1来获得结束位置
        shared_ptr<math_expr_token> parent_data_blk_id_expr_token(new math_expr_token(parent_data_blk_id_token->run() + "+1"));

        // 当前并行级别的数据块的起始位置所在的metadata索引名
        string indices_name_of_blk_offset = "";
        if (pos == THREAD_META)
        {
            indices_name_of_blk_offset = "first_BMT_indices";
        }
        else if (pos == WARP_META)
        {
            indices_name_of_blk_offset = "first_BMW_indices";
        }
        else
        {
            assert(false);
        }

        // 使用这个索引来获得当前块在父块中的偏移量，这个偏移量用作循环的起始位置
        end_loop_var_name = this->generate_fused_memory_access(parent_pos, indices_name_of_blk_offset, parent_data_blk_id_expr_token);
    }

    // 获得步长，步长和是不是最外层索引，以及其父块是什么并行级别的有关，根据线程网格中，当前并行级别的执行单元在父并行级别中的数量来决定
    shared_ptr<var_name_token> step_var_name = NULL;

    if (pos == TBLOCK_META)
    {
        step_var_name = this->BMTB_step;
    }
    else if (pos == WARP_META)
    {
        step_var_name = this->BMW_step;
    }
    else if (pos == THREAD_META)
    {
        step_var_name = this->BMT_step;
    }

    // 创造一个基础的元数据获得的token
    shared_ptr<metadata_get_basic_token> metadata_get_ptr(new metadata_get_basic_token(pos));

    // 用上述的所有内容组合成一个for token，然后输出对应的token
    shared_ptr<for_token> return_token(new for_token(data_type_of_iter_var_token, iter_var_name, begin_loop_var_name, end_loop_var_name,
                                                     step_var_name, pos, metadata_get_ptr, child_for_token, NULL, NULL));

    assert(return_token->static_check() == true);

    return return_token;
}

void code_generator::open_spec_level_of_paral(POS_TYPE pos)
{
    assert(check_pos_type(pos) == true);

    if (pos == TBLOCK_META)
    {
        this->need_tblock_level_paral = true;
    }
    else if (pos == WARP_META)
    {
        this->need_warp_level_paral = true;
    }
    else if (pos == THREAD_META)
    {
        this->need_thread_level_paral = true;
    }
    else
    {
        cout << "code_generator::open_spec_level_of_paral: paral level is illegal" << endl;
        assert(false);
    }
}

// void code_generator::use_relative_nz_index_in_spec_paral(POS_TYPE pos)
// {
//     assert(check_pos_type(pos) == true);

//     if (pos == WARP_META)
//     {
//         this->nz_relative_in_warp_level = true;
//     }
//     else if (pos == THREAD_META)
//     {
//         this->nz_relative_in_thread_level = true;
//     }
//     else
//     {
//         cout << "code_generator::use_relative_nz_index_in_spec_paral: paral level is illegal" << endl;
//         assert(false);
//     }
// }

// void code_generator::use_relative_row_index_in_spec_paral(POS_TYPE pos)
// {
//     assert(check_pos_type(pos) == true);

//     if (pos == WARP_META)
//     {
//         this->row_relative_in_warp_level = true;
//     }
//     else if (pos == THREAD_META)
//     {
//         this->row_relative_in_thread_level = true;
//     }
//     else
//     {
//         cout << "code_generator::use_relative_row_index_in_spec_paral: paral level is illegal" << endl;
//         assert(false);
//     }
// }

// bool code_generator::relative_nz_index_is_used_in_spec_paral(POS_TYPE pos)
// {
//     assert(check_pos_type(pos) == true);

//     if (pos == WARP_META)
//     {
//         return this->nz_relative_in_warp_level;
//     }

//     if (pos == THREAD_META)
//     {
//         return this->nz_relative_in_thread_level;
//     }

//     cout << "code_generator::relative_nz_index_is_used_in_spec_paral: paral level is illegal" << endl;
//     assert(false);

//     return false;
// }

// bool code_generator::relative_row_index_is_used_in_spec_paral(POS_TYPE pos)
// {
//     assert(check_pos_type(pos) == true);

//     if (pos == WARP_META)
//     {
//         return this->row_relative_in_warp_level;
//     }

//     if (pos == THREAD_META)
//     {
//         return this->row_relative_in_thread_level;
//     }

//     cout << "code_generator::relative_row_index_is_used_in_spec_paral: paral level is illegal" << endl;
//     assert(false);

//     return false;
// }

// shared_ptr<var_name_token> code_generator::code_of_thread_grid_info_in_spec_paral(POS_TYPE child_pos, POS_TYPE parent_pos)
// {
//     assert(check_pos_type(child_pos) == true);
//     assert(check_pos_type(parent_pos) == true);
//     assert(former_is_parent_of_latter(parent_pos, child_pos) == true);

//     string var_name = "";

//     if (parent_pos == GLOBAL_META)
//     {
//         if (child_pos == TBLOCK_META)
//         {
//             var_name = this->total_thread_block_num_code();
//         }
//         else if (child_pos == WARP_META)
//         {
//             var_name = this->total_warp_num_code();
//         }
//         else if (child_pos == THREAD_META)
//         {
//             var_name = this->total_thread_num_code();
//         }
//         else
//         {
//             cout << "code_generator::code_of_thread_grid_info_in_spec_paral: illegal child paral level" << endl;
//             assert(false);
//         }
//     }
//     else if (parent_pos == TBLOCK_META)
//     {
//         if (child_pos == WARP_META)
//         {
//             var_name = this->warp_num_in_thread_block_code();
//         }
//         else if (child_pos == THREAD_META)
//         {
//             var_name = this->thread_num_in_thread_block_code();
//         }
//         else
//         {
//             cout << "code_generator::code_of_thread_grid_info_in_spec_paral: illegal child paral level" << endl;
//             assert(false);
//         }
//     }
//     else if (parent_pos == WARP_META)
//     {
//         if (child_pos == THREAD_META)
//         {
//             var_name = this->vector_width_code();
//         }
//         else
//         {
//             cout << "code_generator::code_of_thread_grid_info_in_spec_paral: illegal child paral level" << endl;
//             assert(false);
//         }
//     }
//     else
//     {
//         cout << "code_generator::code_of_thread_grid_info_in_spec_paral: illegal parent paral level" << endl;
//         assert(false);
//     }

//     // 注意这里，如果看每一个warp的thread数量，那么是一个常数
//     if (parent_pos == WARP_META && child_pos == THREAD_META)
//     {
//         shared_ptr<var_name_token> return_token(new var_name_token(var_name, CONSTANT_VAR_TYPE));
//         return return_token;
//     }
//     else
//     {
//         shared_ptr<var_name_token> return_token(new var_name_token(var_name, REGISTER_VAR_TYPE));
//         return return_token;
//     }

//     assert(false);
// }

bool code_generator::shared_mem_is_exist(string shared_mem_name)
{
    // 查看之前是不是登记过对应名字的共享内存
    if (count(this->needed_shared_mem_name_vec.begin(), this->needed_shared_mem_name_vec.end(), shared_mem_name) == 1)
    {
        return true;
    }

    // 除此之外大小只能是0
    assert(count(this->needed_shared_mem_name_vec.begin(), this->needed_shared_mem_name_vec.end(), shared_mem_name) == 0);

    return false;
}

void code_generator::add_new_use_of_shared_mem(data_type shared_mem_data_type, string shared_mem_name, unsigned int shared_mem_size)
{
    assert(check_data_type(shared_mem_data_type) == true);

    // 所占据的大小不能超额
    if (byte_num_of_data_type(shared_mem_data_type) * shared_mem_size >= get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
    {
        cout << "code_generator::add_new_use_of_shared_mem: the usage of shared memory is out of the boundary" << endl;
        assert(false);
    }

    // shared memory的声明不能重复
    if (this->shared_mem_is_exist(shared_mem_name) == true)
    {
        cout << "code_generator::add_new_use_of_shared_mem: name of shared memory are repeated" << endl;
        assert(false);
    }

    // 增加新的内容
    this->needed_shared_mem_data_type_vec.push_back(shared_mem_data_type);
    this->needed_shared_mem_name_vec.push_back(shared_mem_name);
    this->needed_shared_mem_size_vec.push_back(shared_mem_size);
}

shared_ptr<for_token> code_generator::generate_for_structure_of_kernel()
{
    assert(this->check() == true);
    // 总要有一个并行层次
    assert(this->need_tblock_level_paral == true || this->need_warp_level_paral == true || this->need_thread_level_paral == true);

    // 声明三个空指针，分别处理三个层次的for token
    shared_ptr<for_token> thread_level_for_token = NULL;
    shared_ptr<for_token> warp_level_for_token = NULL;
    shared_ptr<for_token> thread_block_level_for_token = NULL;

    // 用一个指针来返回最外层的指针
    shared_ptr<for_token> return_for_token = NULL;

    // 从内到外处理
    if (this->need_thread_level_paral == true)
    {
        // 查看是不是最外层的循环
        bool outest_loop_level = false;

        if (this->need_warp_level_paral == false && this->need_tblock_level_paral == false)
        {
            outest_loop_level = true;
        }

        thread_level_for_token = generate_for_token_of_spec_paral_level(THREAD_META, NULL, outest_loop_level);

        if (outest_loop_level == true)
        {
            return_for_token = thread_level_for_token;
        }
    }

    if (this->need_warp_level_paral == true)
    {
        // 查看是不是最外层的循环
        bool outest_loop_level = false;

        if (this->need_tblock_level_paral == false)
        {
            outest_loop_level = true;
        }

        warp_level_for_token = generate_for_token_of_spec_paral_level(WARP_META, thread_level_for_token, outest_loop_level);

        if (outest_loop_level == true)
        {
            return_for_token = warp_level_for_token;
        }
    }

    if (this->need_tblock_level_paral == true)
    {
        if (this->need_warp_level_paral == true)
        {
            thread_block_level_for_token = generate_for_token_of_spec_paral_level(TBLOCK_META, warp_level_for_token, true);
        }
        else
        {
            thread_block_level_for_token = generate_for_token_of_spec_paral_level(TBLOCK_META, thread_level_for_token, true);
        }

        return_for_token = thread_block_level_for_token;
    }

    assert(return_for_token != NULL);
    assert(return_for_token->static_check() == true);

    return return_for_token;
}

void code_generator::generate_for_structure_of_kernel_and_set_root_for_token()
{
    // 当前的跟root肯定是空的
    assert(this->root_for_token_ptr == NULL);

    this->root_for_token_ptr = this->generate_for_structure_of_kernel();
}

void code_generator::insert_reduction_token_to_for_structure()
{
    // 根据reduction的类型，将归约代码插入到对应的位置
    // 首先是根部必须存在
    assert(this->root_for_token_ptr != NULL);

    // 查看当前的归约层次是不是存在
    if (this->thread_level_reduction_token_ptr != NULL)
    {
        assert(this->thread_level_reduction_token_ptr->get_token_position() == THREAD_META);
        this->insert_reduction_token_to_spec_paral_of_for_structure(THREAD_META);
    }

    if (this->warp_level_reduction_token_ptr != NULL)
    {
        assert(this->warp_level_reduction_token_ptr->get_token_position() == WARP_META);
        this->insert_reduction_token_to_spec_paral_of_for_structure(WARP_META);
    }

    if (this->thread_block_level_reduction_token_ptr != NULL)
    {
        assert(this->thread_block_level_reduction_token_ptr->get_token_position() == TBLOCK_META);
        this->insert_reduction_token_to_spec_paral_of_for_structure(TBLOCK_META);
    }
}

void code_generator::insert_metadata_get_token_to_for_structure()
{
    assert(pos_of_fused_metadata_vec.size() == real_name_of_fused_metadata_vec.size());
    assert(real_name_of_fused_metadata_vec.size() == sub_matrix_id_of_fused_metadata_vec.size());
    assert(sub_matrix_id_of_fused_metadata_vec.size() == access_index_fused_metadata_vec.size());

    // 从thread级别开始，一点点给出元数据获取的token
    // 查看所有的数据依赖，是不是数据依赖存在，那么并行层次就一定存在
    set<POS_TYPE> pos_set;

    // 遍历所有的数据依赖，记录需要的位置
    for (int i = 0; i < pos_of_fused_metadata_vec.size(); i++)
    {
        pos_set.insert(pos_of_fused_metadata_vec[i]);
    }

    if (pos_set.count(THREAD_META) == 1)
    {
        if (this->need_thread_level_paral == false)
        {
            cout << "code_generator::insert_metadata_get_token_to_for_structure: need thead level metadata but do not include corresponding paral" << endl;
            assert(false);
        }
    }

    if (pos_set.count(WARP_META) == 1)
    {
        if (this->need_warp_level_paral == false)
        {
            cout << "code_generator::insert_metadata_get_token_to_for_structure: need warp level metadata but do not include corresponding paral" << endl;
            assert(false);
        }
    }

    if (pos_set.count(TBLOCK_META) == 1)
    {
        if (this->need_tblock_level_paral == false)
        {
            cout << "code_generator::insert_metadata_get_token_to_for_structure: need thread block level metadata but do not include corresponding paral" << endl;
            assert(false);
        }
    }

    // 从thread开始插入
    if (pos_set.count(THREAD_META) == 1)
    {
        // 执行对应位置的元数据插入
        insert_metadata_get_token_to_spec_paral_of_for_structure(THREAD_META);
    }

    if (pos_set.count(WARP_META) == 1)
    {
        insert_metadata_get_token_to_spec_paral_of_for_structure(WARP_META);
    }

    if (pos_set.count(TBLOCK_META) == 1)
    {
        insert_metadata_get_token_to_spec_paral_of_for_structure(TBLOCK_META);
    }
}

void code_generator::insert_reduction_token_to_spec_paral_of_for_structure(POS_TYPE pos)
{
    assert(check_pos_type(pos) == true);

    // 当前的跟for循环是存在的
    assert(this->root_for_token_ptr != NULL);

    // 对应级别的并行层次是存在的
    if (pos == THREAD_META)
    {
        assert(this->need_thread_level_paral == true);
    }

    if (pos == WARP_META)
    {
        assert(this->need_warp_level_paral == true);
    }

    if (pos == TBLOCK_META)
    {
        assert(this->need_tblock_level_paral == true);
    }

    // 遍历各个层次的循环
    shared_ptr<for_token> cur_for_token = this->root_for_token_ptr;

    // 查看当前是不是存在对应并行层次的for token
    bool for_token_in_spec_paral = false;

    while (cur_for_token != NULL)
    {
        // 查看当前for的并行粒度是不是要被插入的
        if (cur_for_token->get_token_position() == pos)
        {
            for_token_in_spec_paral = true;

            // 首先查看是不是存在对应的reduction，肯定是不存在的
            assert(cur_for_token->get_reduction_token() == NULL);

            // 执行赋值操作
            // 不同的并行层次的并行插入不同的归约方法
            if (pos == THREAD_META)
            {
                assert(this->thread_level_reduction_token_ptr != NULL);
                cur_for_token->set_reduction_code(this->thread_level_reduction_token_ptr);
                cur_for_token->set_glue_code_block(this->glue_code_in_thread_level_ptr);
                break;
            }
            else if (pos == WARP_META)
            {
                assert(this->warp_level_reduction_token_ptr != NULL);
                cur_for_token->set_reduction_code(this->warp_level_reduction_token_ptr);
                cur_for_token->set_glue_code_block(this->glue_code_in_warp_level_ptr);
                break;
            }
            else if (pos == TBLOCK_META)
            {
                assert(this->thread_block_level_reduction_token_ptr != NULL);
                cur_for_token->set_reduction_code(this->thread_block_level_reduction_token_ptr);
                cur_for_token->set_glue_code_block(this->glue_code_in_thread_block_level_ptr);
                break;
            }
            else
            {
                assert(false);
            }
        }

        cur_for_token = dynamic_pointer_cast<for_token>(cur_for_token->get_child_for_token());
    }

    // 必然找到了对应的层次的for
    assert(for_token_in_spec_paral == true);
}

// 将对应的代码插入对应的位置
void code_generator::insert_metadata_get_token_to_spec_paral_of_for_structure(POS_TYPE pos)
{
    assert(check_pos_type(pos) == true);

    // 当前的跟for循环是存在的
    assert(this->root_for_token_ptr != NULL);

    // 生成之前必须查看对应层次的并行是不是存在
    if (pos == THREAD_META)
    {
        assert(this->need_thread_level_paral == true);
    }
    else if (pos == WARP_META)
    {
        assert(this->need_warp_level_paral == true);
    }
    else if (pos == TBLOCK_META)
    {
        assert(this->need_tblock_level_paral == true);
    }
    else
    {
        assert(false);
    }

    // 获取对应的metadata get token
    shared_ptr<metadata_set_get_token> metadata_get_token_ptr = this->generate_token_of_fused_metadata_get_in_spec_paral_level(pos);

    // 偷偷执行一次，登记数据依赖
    metadata_get_token_ptr->run();

    // 查看并且检查
    assert(metadata_get_token_ptr->static_check() == true);

    // 遍历各个层次的循环
    shared_ptr<for_token> cur_for_token = this->root_for_token_ptr;

    // 查看当前是不是存在对应并行层次的for token
    bool for_token_in_spec_paral = false;

    // 找到对应层次的循环结构
    while (cur_for_token != NULL)
    {
        // 查看当前for的并行粒度是不是要被插入的
        if (cur_for_token->get_token_position() == pos)
        {
            for_token_in_spec_paral = true;

            // 之前产生不会是空的
            assert(cur_for_token->get_metadata_get_token() != NULL);

            assert(cur_for_token->get_metadata_get_token()->get_token_position() == pos);

            // 执行赋值操作
            // 不同的并行层次的并行插入不同的归约方法
            cur_for_token->set_metadata_get_code(metadata_get_token_ptr);

            break;
        }

        cur_for_token = dynamic_pointer_cast<for_token>(cur_for_token->get_child_for_token());
    }

    // 必然找到了对应的层次的for
    assert(for_token_in_spec_paral == true);
}

bool code_generator::reduction_token_is_existing(POS_TYPE pos)
{
    assert(check_pos_type(pos) == true);

    if (pos == GLOBAL_META)
    {
        if (this->global_level_reduction_token_ptr != NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (pos == TBLOCK_META)
    {
        if (this->thread_block_level_reduction_token_ptr != NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (pos == WARP_META)
    {
        if (this->warp_level_reduction_token_ptr != NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (pos == THREAD_META)
    {
        if (this->thread_level_reduction_token_ptr != NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    assert(false);
    return false;
}

void code_generator::set_reduction_token(POS_TYPE pos, shared_ptr<reduction_basic_token> token_ptr)
{
    assert(check_pos_type(pos) == true);
    assert(token_ptr != NULL);
    assert(token_ptr->static_check() == true);
    // 对应级别的归约还不存在
    assert(this->reduction_token_is_existing(pos) == false);
    // 插入位置和插入token的pos类型是一样的
    assert(pos == token_ptr->get_token_position());

    if (pos == GLOBAL_META)
    {
        assert(this->global_level_reduction_token_ptr == NULL);
        this->global_level_reduction_token_ptr = token_ptr;
    }
    else if (pos == TBLOCK_META)
    {
        assert(this->thread_block_level_reduction_token_ptr == NULL);
        this->thread_block_level_reduction_token_ptr = token_ptr;
    }
    else if (pos == WARP_META)
    {
        assert(this->warp_level_reduction_token_ptr == NULL);
        this->warp_level_reduction_token_ptr = token_ptr;
    }
    else if (pos == THREAD_META)
    {
        assert(this->thread_level_reduction_token_ptr == NULL);
        this->thread_level_reduction_token_ptr = token_ptr;
    }
    else
    {
        assert(false);
    }
}

string code_generator::code_of_root_for_token()
{
    assert(this->root_for_token_ptr != NULL);
    assert(this->root_for_token_ptr->static_check() == true);

    // 将for循环打印出来
    return this->root_for_token_ptr->run();
}

string code_generator::code_of_for_structure()
{
    // 首先是查看原料是不是都存在
    assert(this->root_for_token_ptr != NULL);
    assert(this->root_for_token_ptr->static_check() == true);

    // 查看对应层次的归约和对应层次的并行是不是存在
    if (this->thread_block_level_reduction_token_ptr != NULL)
    {
        assert(this->need_tblock_level_paral == true);
    }

    if (this->warp_level_reduction_token_ptr != NULL)
    {
        assert(this->need_warp_level_paral == true);
    }

    if (this->thread_level_reduction_token_ptr != NULL)
    {
        assert(this->need_thread_level_paral == true);
    }
    
    this->generate_glue_code();

    // 将reduction插入
    this->insert_reduction_token_to_for_structure();

    // 将元数据备份下来
    back_up_of_metadata_register_t back_up = this->backup_metatdata_and_global_var();

    // 先打印一次
    this->code_of_root_for_token();

    // 然后插入metadata的内容
    this->insert_metadata_get_token_to_for_structure();

    this->recover_metatdata_and_global_var(back_up);


    // 最后打印出输出
    return this->code_of_root_for_token();
}

back_up_of_metadata_register_t code_generator::backup_metatdata_and_global_var()
{
    back_up_of_metadata_register_t return_back_up;

    return_back_up.pos_of_needed_metadata_vec = this->pos_of_needed_metadata_vec;
    return_back_up.real_name_of_needed_metadata_vec = this->real_name_of_needed_metadata_vec;
    return_back_up.sub_matrix_id_of_needed_metadata_vec = this->sub_matrix_id_of_needed_metadata_vec;

    return_back_up.added_metadata_dependency = this->added_metadata_dependency;

    return_back_up.pos_of_fused_metadata_vec = this->pos_of_fused_metadata_vec;
    return_back_up.real_name_of_fused_metadata_vec = this->real_name_of_fused_metadata_vec;
    return_back_up.sub_matrix_id_of_fused_metadata_vec = this->sub_matrix_id_of_fused_metadata_vec;

    return_back_up.access_index_fused_metadata_vec = this->access_index_fused_metadata_vec;
    return_back_up.global_var_init_token_vec = this->global_var_init_token_vec;

    return_back_up.needed_shared_mem_data_type_vec = this->needed_shared_mem_data_type_vec;
    return_back_up.needed_shared_mem_name_vec = this->needed_shared_mem_name_vec;
    return_back_up.needed_shared_mem_size_vec = this->needed_shared_mem_size_vec;

    return return_back_up;
}

void code_generator::recover_metatdata_and_global_var(back_up_of_metadata_register_t back_up)
{
    this->pos_of_needed_metadata_vec = back_up.pos_of_needed_metadata_vec;
    this->real_name_of_needed_metadata_vec = back_up.real_name_of_needed_metadata_vec;
    this->sub_matrix_id_of_needed_metadata_vec = back_up.sub_matrix_id_of_needed_metadata_vec;

    this->added_metadata_dependency = back_up.added_metadata_dependency;

    this->pos_of_fused_metadata_vec = back_up.pos_of_fused_metadata_vec;
    this->real_name_of_fused_metadata_vec = back_up.real_name_of_fused_metadata_vec;
    this->sub_matrix_id_of_fused_metadata_vec = back_up.sub_matrix_id_of_fused_metadata_vec;

    this->access_index_fused_metadata_vec = back_up.access_index_fused_metadata_vec;
    this->global_var_init_token_vec = back_up.global_var_init_token_vec;

    this->needed_shared_mem_data_type_vec = back_up.needed_shared_mem_data_type_vec;
    this->needed_shared_mem_name_vec = back_up.needed_shared_mem_name_vec;
    this->needed_shared_mem_size_vec = back_up.needed_shared_mem_size_vec;

    // 恢复完之后必须保证正确
    assert(this->check() == true);
}

void code_generator::compile_and_set_for_code()
{
    if (this->root_for_token_ptr != NULL)
    {
        cout << "code_generator::compile_and_set_for_code: the for structure has already existed" << endl;
        assert(false);
    }

    // 处理for循环结构
    this->generate_for_structure_of_kernel_and_set_root_for_token();

    // 先检查之前有没有编译过
    if (this->compiled_for_code != "")
    {
        cout << "code_generator::compile_and_set_for_code: the code has already compiled" << endl;
        assert(false);
    }

    stringstream ss;

    ss << this->code_of_for_structure() << endl;

    if (this->global_level_reduction_token_ptr != NULL)
    {
        assert(this->global_level_reduction_token_ptr->static_check() == true);
        ss << this->global_level_reduction_token_ptr->run() << endl;
    }

    // 创建for循环的内容
    this->compiled_for_code = ss.str();
}

bool code_generator::if_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id)
{
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    if (target_array->get_data_type() == FLOAT || target_array->get_data_type() == DOUBLE || get_config()["MODEL_DRIVEN_COMPRESS"].as_bool() == false)
    {
        return false;
    }
    unsigned long item1 = target_array->read_integer_from_arr(0);
    unsigned long item2 = target_array->read_integer_from_arr(1);
    unsigned long coef = item2 - item1;

    for (unsigned long i = 0; i < target_array->get_len() - 1; i++)
    {
        item1 = target_array->read_integer_from_arr(i);
        item2 = target_array->read_integer_from_arr(i + 1);
        if ((item2 - item1) != coef)
        {
            return false;
        }
    }

    return true;
}

bool code_generator::if_branch_compress(POS_TYPE pos, string real_name, int sub_matrix_id)
{

    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    if (target_array->get_data_type() == FLOAT || target_array->get_data_type() == DOUBLE || get_config()["MODEL_DRIVEN_COMPRESS"].as_bool() == false)
    {
        return false;
    }
    unsigned long item = target_array->read_integer_from_arr(0);
    unsigned long count = 1;

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long cur_val = target_array->read_integer_from_arr(i);

        if (cur_val != item)
        {
            count += 1;
            item = cur_val;

            if (count >= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer())
            {
                return false;
            }
        }
    }

    return true;
}

bool code_generator::if_cycle_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id)
{

    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    if (target_array->get_data_type() == FLOAT || target_array->get_data_type() == DOUBLE || get_config()["MODEL_DRIVEN_COMPRESS"].as_bool() == false)
    {
        return false;
    }
    unsigned long item1 = target_array->read_integer_from_arr(0);
    unsigned long item2 = target_array->read_integer_from_arr(1);
    unsigned long new_coefficient = item2 - item1;
    unsigned long new_intercept = item1;
    unsigned long cycle_num = 1;

    for (unsigned long i = 1; i < target_array->get_len(); i++)
    {
        unsigned long cur = target_array->read_integer_from_arr(i);
        if (cur == new_intercept)
        {
            cycle_num = i;
        }
    }

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long item_val = target_array->read_integer_from_arr(i);
        unsigned long index_inner_cycle = i % cycle_num;

        if (index_inner_cycle == 0)
        {
            if (item_val != new_intercept)
            {
                return false;
            }
        }

        if (index_inner_cycle != 0 && (item_val - new_intercept) / index_inner_cycle != new_coefficient)
        {
            return false;
        }
    }

    return true;
}

bool code_generator::if_cycle_increase_compress(POS_TYPE pos, string real_name, int sub_matrix_id)
{
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    if (target_array->get_data_type() == FLOAT || target_array->get_data_type() == DOUBLE || get_config()["MODEL_DRIVEN_COMPRESS"].as_bool() == false)
    {
        return false;
    }
    unsigned long item1 = target_array->read_integer_from_arr(0);
    unsigned long new_cycle_num = 1;

    for (unsigned long i = 1; i < target_array->get_len(); i++)
    {
        unsigned long cur_element = target_array->read_integer_from_arr(i);
        if (cur_element != item1)
        {
            if (cur_element < item1)
            {
                return false;
            }
            new_cycle_num = i;
            break;
        }
    }

    if (target_array->get_len() % new_cycle_num != 0)
    {
        return false;
    }

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long cur_element = target_array->read_integer_from_arr(i);
        unsigned long cycle_id = (unsigned long)(i / new_cycle_num);
        if (cycle_id != 0)
        {
            if ((cur_element - item1) % cycle_id != 0)
            {
                return false;
            }
        }
    }

    return true;
}

bool code_generator::if_residual_compress(POS_TYPE pos, string real_name, int sub_matrix_id)
{
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    if (target_array->get_data_type() == FLOAT || target_array->get_data_type() == DOUBLE || get_config()["MODEL_DRIVEN_COMPRESS"].as_bool() == false)
    {
        return false;
    }
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    double a, b;
    double n = target_array->get_len();

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long item = target_array->read_integer_from_arr(i);
        t1 += i * i;
        t2 += i;
        t3 += i * item;
        t4 += item;
    }
    a = (t3 * n - t2 * t4) / (t1 * n - t2 * t2);
    b = (t1 * t4 - t2 * t3) / (t1 * n - t2 * t2);

    long aa = a;
    long bb = b;

    unsigned long max_item_1 = 0;
    unsigned long max_item_2 = 0;

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        long error;
        error = target_array->read_integer_from_arr(i) - aa * i - bb;
        if (error >= 0)
        {
            if (error > max_item_1)
            {
                max_item_1 = error;
            }
        }
        else
        {
            if (-error > max_item_2)
            {
                max_item_2 = -error;
            }
        }
    }

    bb -= max_item_2;

    unsigned long total_max_item = max_item_1 + max_item_2;
    data_type type_ori = target_array->get_compress_data_type();
    data_type type_compress = find_most_suitable_data_type(total_max_item);

    if (type_compress < type_ori)
    {
        real_name = real_name + "_res";
        string item_name = get_metadata_item_name(pos, real_name, sub_matrix_id);
        if (this->meta_data_set_ptr->is_exist(item_name) == true)
        {
        }
        else
        {
            vector<unsigned long> array_error;
            for (unsigned long i = 0; i < target_array->get_len(); i++)
            {
                unsigned long new_item = target_array->read_integer_from_arr(i) - aa * i - bb;
                array_error.push_back(new_item);
            }

            shared_ptr<universal_array> array_error_ptr(new universal_array(&(array_error[0]), array_error.size(), UNSIGNED_LONG));
            shared_ptr<meta_data_item> new_data_item(new meta_data_item(array_error_ptr, pos, real_name, sub_matrix_id));
            this->meta_data_set_ptr->add_element(new_data_item);
        }

        return true;
    }
    else
    {
        return false;
    }
}

shared_ptr<basic_token> code_generator::get_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr)
{
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    unsigned long item1 = target_array->read_integer_from_arr(0);
    unsigned long item2 = target_array->read_integer_from_arr(1);
    unsigned long coef = item2 - item1;

    string result_string;
    if (coef == 0)
    {
        result_string = to_string(item1);
    }
    else if (coef != 1)
    {
        result_string = to_string(coef) + " * (" + access_index_expr->run() + ")";
    }
    else
    {
        result_string = access_index_expr->run();
    }

    if (item1 != 0)
    {
        result_string += " + " + to_string(item1);
    }

    shared_ptr<math_expr_token> compressed_result_math(new math_expr_token(result_string));
    shared_ptr<var_assign_token> compressed_result(new var_assign_token(var_token, compressed_result_math));
    shared_ptr<basic_token> return_result = compressed_result;
    return return_result;
}

shared_ptr<basic_token> code_generator::get_branch_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr)
{
    vector<unsigned long> new_low_bound_vec;
    vector<unsigned long> new_up_bound_vec;
    vector<unsigned long> new_val_vec;
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    unsigned long item = target_array->read_integer_from_arr(0);

    new_low_bound_vec.push_back(0);
    new_val_vec.push_back(item);

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long cur_val = target_array->read_integer_from_arr(i);

        if (cur_val != new_val_vec[new_val_vec.size() - 1])
        {
            new_up_bound_vec.push_back(i - 1);
            new_low_bound_vec.push_back(i);
            new_val_vec.push_back(cur_val);
        }
    }
    new_up_bound_vec.push_back(target_array->get_len() - 1);

    vector<shared_ptr<math_expr_token>> condition;
    vector<unsigned int> ptr;
    vector<shared_ptr<var_assign_token>> assign;
    for (unsigned long branch_index = 0; branch_index < new_low_bound_vec.size(); branch_index++)
    {
        unsigned long low_bound = new_low_bound_vec[branch_index];
        unsigned long up_bound = new_up_bound_vec[branch_index];
        unsigned long con_val = new_val_vec[branch_index];

        assert(low_bound <= up_bound);

        if (low_bound == up_bound)
        {
            string condition_str;
            condition_str = access_index_expr->run() + " == " + to_string(low_bound);
            shared_ptr<math_expr_token> condition_token(new math_expr_token(condition_str));
            condition.push_back(condition_token);
        }
        else
        {
            string condition_str;
            condition_str = access_index_expr->run() + " >= " + to_string(low_bound) + " && " + access_index_expr->run() + " <= " + to_string(up_bound);
            shared_ptr<math_expr_token> condition_token(new math_expr_token(condition_str));
            condition.push_back(condition_token);
        }
        ptr.push_back(1);
        shared_ptr<math_expr_token> branch_result(new math_expr_token(to_string(con_val)));
        shared_ptr<var_assign_token> assign_expr(new var_assign_token(var_token, branch_result));
        assign.push_back(assign_expr);
    }
    assert(condition.size() == ptr.size() && ptr.size() == assign.size());

    shared_ptr<basic_token> return_result;
    shared_ptr<if_else_token> if_token_as_result(new if_else_token(condition, ptr, assign));

    return_result = if_token_as_result;

    return return_result;
}

shared_ptr<basic_token> code_generator::get_cycle_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr)
{
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    unsigned long item1 = target_array->read_integer_from_arr(0);
    unsigned long item2 = target_array->read_integer_from_arr(1);
    unsigned long new_coefficient = item2 - item1;
    unsigned long new_intercept = item1;
    unsigned long cycle_num = 0;

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long cur = target_array->read_integer_from_arr(i);
        if (cur == new_intercept)
        {
            cycle_num = i;
        }
    }

    string result_string;
    result_string = "( (" + access_index_expr->run() + ") % " + to_string(cycle_num) + " ) * " + to_string(new_coefficient);
    if (new_intercept != 0)
    {
        result_string += " + " + to_string(new_intercept);
    }

    shared_ptr<math_expr_token> compressed_result_math(new math_expr_token(result_string));
    shared_ptr<var_assign_token> compressed_result(new var_assign_token(var_token, compressed_result_math));
    shared_ptr<basic_token> return_result = compressed_result;
    return return_result;
}

shared_ptr<basic_token> code_generator::get_cycle_increase_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr)
{

    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    unsigned long item1 = target_array->read_integer_from_arr(0);
    unsigned long new_intercept = item1;
    unsigned long new_cycle_num = 0;
    unsigned long k = 0;

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long cur_element = target_array->read_integer_from_arr(i);
        if (cur_element != item1)
        {
            new_cycle_num = i;
            k = cur_element - item1;
            break;
        }
    }

    string result_string;
    result_string = "( (" + access_index_expr->run() + ") / " + to_string(new_cycle_num) + " ) * " + to_string(k);
    if (new_intercept != 0)
    {
        result_string += " + " + to_string(new_intercept);
    }

    shared_ptr<math_expr_token> compressed_result_math(new math_expr_token(result_string));
    shared_ptr<var_assign_token> compressed_result(new var_assign_token(var_token, compressed_result_math));
    shared_ptr<basic_token> return_result = compressed_result;
    return return_result;
}

shared_ptr<basic_token> code_generator::get_residual_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr)
{
    shared_ptr<universal_array> target_array = this->meta_data_set_ptr->get_element(pos, real_name, sub_matrix_id)->get_metadata_arr();
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    double a, b;
    double n = target_array->get_len();

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        unsigned long item = target_array->read_integer_from_arr(i);
        t1 += i * i;
        t2 += i;
        t3 += i * item;
        t4 += item;
    }
    a = (t3 * n - t2 * t4) / (t1 * n - t2 * t2);
    b = (t1 * t4 - t2 * t3) / (t1 * n - t2 * t2);

    long aa = a;
    long bb = b;

    unsigned long max_item_1 = 0;
    unsigned long max_item_2 = 0;

    for (unsigned long i = 0; i < target_array->get_len(); i++)
    {
        long error;
        error = target_array->read_integer_from_arr(i) - aa * i - bb;
        if (error >= 0)
        {
            if (error > max_item_1)
            {
                max_item_1 = error;
            }
        }
        else
        {
            if (-error > max_item_2)
            {
                max_item_2 = -error;
            }
        }
    }

    bb -= max_item_2;
    real_name = real_name + "_res";

    string result_string;
    string mem_access_arr_name = get_metadata_item_name(pos, real_name, sub_matrix_id);
    shared_ptr<var_name_token> mem_access_arr_name_token(new var_name_token(mem_access_arr_name, GLOBAL_MEM_VAR_TYPE));
    shared_ptr<arr_access_token> mem_access_token(new arr_access_token(var_token, mem_access_arr_name_token, access_index_expr->run()));

    result_string = to_string(aa) + " * (" + access_index_expr->run() + ") + (" + to_string(bb) + ")";

    shared_ptr<math_expr_token> compressed_result_math(new math_expr_token(result_string, mem_access_token, "+"));
    shared_ptr<var_assign_token> compressed_result(new var_assign_token(var_token, compressed_result_math));
    shared_ptr<basic_token> return_result = compressed_result;
    return return_result;
}

shared_ptr<basic_token> code_generator::get_compress_and_relative_result(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr)
{
    if (if_linear_compress(pos, real_name, sub_matrix_id) == true && (this->if_dependency_exist(pos, real_name + "_relative_to_BMW", sub_matrix_id) == false && this->if_dependency_exist(pos, real_name + "_relative_to_BMTB", sub_matrix_id) == false))
    {
        return get_linear_compress(pos, real_name, sub_matrix_id, var_token, access_index_expr);
    }
    else if (if_branch_compress(pos, real_name, sub_matrix_id) == true && (this->if_dependency_exist(pos, real_name + "_relative_to_BMW", sub_matrix_id) == false && this->if_dependency_exist(pos, real_name + "_relative_to_BMTB", sub_matrix_id) == false))
    {
        return get_branch_compress(pos, real_name, sub_matrix_id, var_token, access_index_expr);
    }
    else if (if_cycle_linear_compress(pos, real_name, sub_matrix_id) == true && (this->if_dependency_exist(pos, real_name + "_relative_to_BMW", sub_matrix_id) == false && this->if_dependency_exist(pos, real_name + "_relative_to_BMTB", sub_matrix_id) == false))
    {
        return get_cycle_linear_compress(pos, real_name, sub_matrix_id, var_token, access_index_expr);
    }
    else if (if_cycle_increase_compress(pos, real_name, sub_matrix_id) == true && (this->if_dependency_exist(pos, real_name + "_relative_to_BMW", sub_matrix_id) == false && this->if_dependency_exist(pos, real_name + "_relative_to_BMTB", sub_matrix_id) == false))
    {
        return get_cycle_increase_compress(pos, real_name, sub_matrix_id, var_token, access_index_expr);
    }
    else if (if_residual_compress(pos, real_name, sub_matrix_id) == true && (this->if_dependency_exist(pos, real_name + "_relative_to_BMW", sub_matrix_id) == false && this->if_dependency_exist(pos, real_name + "_relative_to_BMTB", sub_matrix_id) == false))
    {
        return get_residual_compress(pos, real_name, sub_matrix_id, var_token, access_index_expr);
    }
    else
    {
        string real_metadata_name_1 = real_name + "_relative_to_BMW";
        string real_metadata_name_2 = real_name + "_relative_to_BMTB";
        POS_TYPE parent_pos;
        bool end_loop = false;
        if (access_index_expr->run().find("+1") != string::npos || access_index_expr->run().find("+ 1") != string::npos)
        {
            end_loop = true;
        }

        if (real_name.find("relative") == string::npos)
        {
            if (this->meta_data_set_ptr->is_exist(pos, real_metadata_name_1, sub_matrix_id) == true)
            {
                parent_pos = WARP_META;

                if (end_loop == false)
                {
                    shared_ptr<math_expr_token> parent_id_token(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run()));
                    string var_of_readed_content_parent = var_of_metadata_from_spec_paral(parent_pos, real_name, sub_matrix_id, parent_id_token);

                    string var_of_readed_content_relative = var_of_metadata_from_spec_paral(pos, real_metadata_name_1, sub_matrix_id, access_index_expr);

                    string result_right_string = var_of_readed_content_parent + " + " + var_of_readed_content_relative;
                    shared_ptr<math_expr_token> result_expr(new math_expr_token(result_right_string));
                    shared_ptr<var_assign_token> result_token(new var_assign_token(var_token, result_expr));
                    return result_token;
                }
                else
                {
                    shared_ptr<math_expr_token> parent_id_token(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run()));
                    shared_ptr<math_expr_token> next_parent_id_token(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run() + "+1"));

                    string indices_name_of_blk_offset = "";
                    if (pos == THREAD_META)
                    {
                        indices_name_of_blk_offset = "first_BMT_indices";
                    }
                    else if (pos == WARP_META)
                    {
                        indices_name_of_blk_offset = "first_BMW_indices";
                    }

                    string var_of_readed_content_indices = var_of_metadata_from_spec_paral(parent_pos, indices_name_of_blk_offset, sub_matrix_id, next_parent_id_token);

                    string var_of_readed_content_parent = var_of_metadata_from_spec_paral(parent_pos, real_name, sub_matrix_id, parent_id_token);
                    string var_of_readed_content_next_parent = var_of_metadata_from_spec_paral(parent_pos, real_name, sub_matrix_id, next_parent_id_token);

                    string var_of_readed_content_relative = var_of_metadata_from_spec_paral(pos, real_metadata_name_1, sub_matrix_id, access_index_expr);

                    string result_right_string_1 = var_of_readed_content_parent + " + " + var_of_readed_content_relative;
                    shared_ptr<math_expr_token> result_expr_1(new math_expr_token(result_right_string_1));

                    string result_right_string_2 = var_of_readed_content_next_parent;
                    shared_ptr<math_expr_token> result_expr_2(new math_expr_token(result_right_string_2));

                    string condition_string = access_index_expr->run() + "<" + var_of_readed_content_indices;
                    shared_ptr<math_expr_token> condition_expr(new math_expr_token(condition_string));

                    shared_ptr<var_assign_token> ass_token_1(new var_assign_token(var_token, result_expr_1));
                    shared_ptr<var_assign_token> ass_token_2(new var_assign_token(var_token, result_expr_2));

                    vector<shared_ptr<math_expr_token>> condition_vec;
                    vector<unsigned int> ptr(2, 1);
                    vector<shared_ptr<var_assign_token>> assign_vec;

                    condition_vec.push_back(condition_expr);
                    assign_vec.push_back(ass_token_1);
                    assign_vec.push_back(ass_token_2);

                    shared_ptr<if_else_token> result_if_token(new if_else_token(condition_vec, ptr, assign_vec));
                    return result_if_token;
                }
            }
            else if (this->meta_data_set_ptr->is_exist(pos, real_metadata_name_2, sub_matrix_id) == true)
            {
                parent_pos = TBLOCK_META;

                if (end_loop == false)
                {
                    shared_ptr<math_expr_token> parent_id_token(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run()));
                    string var_of_readed_content_parent = var_of_metadata_from_spec_paral(parent_pos, real_name, sub_matrix_id, parent_id_token);

                    string var_of_readed_content_relative = var_of_metadata_from_spec_paral(pos, real_metadata_name_2, sub_matrix_id, access_index_expr);

                    string result_right_string = var_of_readed_content_parent + " + " + var_of_readed_content_relative;
                    shared_ptr<math_expr_token> result_expr(new math_expr_token(result_right_string));
                    shared_ptr<var_assign_token> result_token(new var_assign_token(var_token, result_expr));
                    return result_token;
                }
                else
                {
                    shared_ptr<math_expr_token> parent_id_token(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run()));
                    shared_ptr<math_expr_token> next_parent_id_token(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(parent_pos)->run() + "+1"));

                    string indices_name_of_blk_offset = "";
                    if (pos == THREAD_META)
                    {
                        indices_name_of_blk_offset = "first_BMT_indices";
                    }
                    else if (pos == WARP_META)
                    {
                        indices_name_of_blk_offset = "first_BMW_indices";
                    }

                    string var_of_readed_content_indices = var_of_metadata_from_spec_paral(parent_pos, indices_name_of_blk_offset, sub_matrix_id, next_parent_id_token);

                    string var_of_readed_content_parent = var_of_metadata_from_spec_paral(parent_pos, real_name, sub_matrix_id, parent_id_token);
                    string var_of_readed_content_next_parent = var_of_metadata_from_spec_paral(parent_pos, real_name, sub_matrix_id, next_parent_id_token);

                    string var_of_readed_content_relative = var_of_metadata_from_spec_paral(pos, real_metadata_name_2, sub_matrix_id, access_index_expr);

                    string result_right_string_1 = var_of_readed_content_parent + " + " + var_of_readed_content_relative;
                    shared_ptr<math_expr_token> result_expr_1(new math_expr_token(result_right_string_1));

                    string result_right_string_2 = var_of_readed_content_next_parent;
                    shared_ptr<math_expr_token> result_expr_2(new math_expr_token(result_right_string_2));

                    string condition_string = access_index_expr->run() + "<" + var_of_readed_content_indices;
                    shared_ptr<math_expr_token> condition_expr(new math_expr_token(condition_string));

                    shared_ptr<var_assign_token> ass_token_1(new var_assign_token(var_token, result_expr_1));
                    shared_ptr<var_assign_token> ass_token_2(new var_assign_token(var_token, result_expr_2));

                    vector<shared_ptr<math_expr_token>> condition_vec;
                    vector<unsigned int> ptr(2, 1);
                    vector<shared_ptr<var_assign_token>> assign_vec;

                    condition_vec.push_back(condition_expr);
                    assign_vec.push_back(ass_token_1);
                    assign_vec.push_back(ass_token_2);

                    shared_ptr<if_else_token> result_if_token(new if_else_token(condition_vec, ptr, assign_vec));
                    return result_if_token;
                }
            }
            else
            {
                string mem_access_arr_name = get_metadata_item_name(pos, real_name, sub_matrix_id);
                shared_ptr<var_name_token> mem_access_arr_name_token(new var_name_token(mem_access_arr_name, GLOBAL_MEM_VAR_TYPE));
                shared_ptr<arr_access_token> mem_access_token(new arr_access_token(var_token, mem_access_arr_name_token, access_index_expr->run()));
                return mem_access_token;
            }
        }
        else
        {
            string mem_access_arr_name = get_metadata_item_name(pos, real_name, sub_matrix_id);
            shared_ptr<var_name_token> mem_access_arr_name_token(new var_name_token(mem_access_arr_name, GLOBAL_MEM_VAR_TYPE));
            shared_ptr<arr_access_token> mem_access_token(new arr_access_token(var_token, mem_access_arr_name_token, access_index_expr->run()));
            return mem_access_token;
        }
    }
}

void code_generator::generate_glue_code()
{
    if (this->thread_level_reduction_token_ptr != NULL)
    {
        if (this->warp_level_reduction_token_ptr != NULL)
        {
            // shared_ptr<basic_glue_code> glue_code;
            // shared_ptr<basic_IO_of_reduction> input = this->thread_level_reduction_token_ptr->get_output_IO();
            // shared_ptr<basic_IO_of_reduction> output = this->warp_level_reduction_token_ptr->get_input_IO();
            // glue_code = this->get_glue_code_token_according_to_input_and_ouput_IO(input, output);

            // this->glue_code_in_thread_level_ptr = glue_code;
        }
        else if (this->thread_block_level_reduction_token_ptr != NULL)
        {
            // shared_ptr<basic_glue_code> glue_code;
            // shared_ptr<basic_IO_of_reduction> input = this->thread_level_reduction_token_ptr->get_output_IO();
            // shared_ptr<basic_IO_of_reduction> output = this->thread_block_level_reduction_token_ptr->get_input_IO();
            // glue_code = this->get_glue_code_token_according_to_input_and_ouput_IO(input, output);
            // this->glue_code_in_thread_level_ptr = glue_code;
        }
        else
        {
            shared_ptr<basic_glue_code> glue_code;
            shared_ptr<basic_IO_of_reduction> input = this->thread_level_reduction_token_ptr->get_output_IO();
            shared_ptr<basic_IO_of_reduction> output = NULL;
            glue_code = this->get_glue_code_token_according_to_input_and_ouput_IO(input, output);
            this->glue_code_in_thread_level_ptr = glue_code;
        }
    }

    if (this->warp_level_reduction_token_ptr != NULL)
    {
        if (this->thread_block_level_reduction_token_ptr != NULL)
        {
            // shared_ptr<basic_glue_code> glue_code;
            // shared_ptr<basic_IO_of_reduction> input = this->warp_level_reduction_token_ptr->get_output_IO();
            // shared_ptr<basic_IO_of_reduction> output = this->thread_block_level_reduction_token_ptr->get_input_IO();
            // glue_code = this->get_glue_code_token_according_to_input_and_ouput_IO(input, output);
            // this->glue_code_in_warp_level_ptr = glue_code;
        }
        else
        {
            // shared_ptr<basic_glue_code> glue_code;
            // shared_ptr<basic_IO_of_reduction> input = this->warp_level_reduction_token_ptr->get_output_IO();
            // shared_ptr<basic_IO_of_reduction> output = NULL;
            // glue_code = this->get_glue_code_token_according_to_input_and_ouput_IO(input, output);
            // this->glue_code_in_warp_level_ptr = glue_code;
        }
    }

    if (this->thread_block_level_reduction_token_ptr != NULL)
    {
        // shared_ptr<basic_glue_code> glue_code;
        // shared_ptr<basic_IO_of_reduction> input = this->thread_block_level_reduction_token_ptr->get_output_IO();
        // shared_ptr<basic_IO_of_reduction> output = NULL;
        // glue_code = this->get_glue_code_token_according_to_input_and_ouput_IO(input, output);
        // this->glue_code_in_thread_block_level_ptr = glue_code;
    }
}
shared_ptr<basic_glue_code> code_generator::get_glue_code_token_according_to_input_and_ouput_IO(shared_ptr<basic_IO_of_reduction> input_of_glue_code, shared_ptr<basic_IO_of_reduction> output_of_glue_code)
{
    if (input_of_glue_code->get_name_of_IO_type() == "one_register_result_IO_of_reduction" && input_of_glue_code->get_generated_pos() == THREAD_META && output_of_glue_code == NULL)
    {

        string glue_string;
        shared_ptr<math_expr_token> BMT_id(new math_expr_token(this->code_of_data_block_id_distributed_to_spec_paral(THREAD_META)->run()));
        vector<shared_ptr<basic_token>> first_row_indices_of_thread = this->generate_unfused_memory_access(THREAD_META, "first_row_indices", BMT_id, true, "first_row_thread");
        string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
        string result_name = input_of_glue_code->var_name_token_of_IO_register()->run();
        for (int i = 0; i < first_row_indices_of_thread.size(); i++)
        {
            glue_string += first_row_indices_of_thread[i]->run();
        }

        if (this->thread_for_row == true)
        {
            if (input_of_glue_code->get_count() > 1)
            {
                glue_string += "for(int factor = 0; factor <" + to_string(input_of_glue_code->get_count()) + ";factor++)\n";
                glue_string += "{\n";
                glue_string += "y_arr[" + row_name + " * K + dense_matrix_ptr + factor] = " + result_name + "[factor];\n";
                glue_string += "}\n";
            }
            else
            {
                glue_string += "y_arr[" + row_name + " * K + dense_matrix_ptr] = " + result_name + "[0];\n";
            }
        }
        else
        {
            if (input_of_glue_code->get_count() > 1)
            {
                glue_string += "for(int factor = 0; factor <" + to_string(input_of_glue_code->get_count()) + ";factor++)\n";
                glue_string += "{\n";
                glue_string += "atomicAdd(&y_arr[" + row_name + " * K + dense_matrix_ptr + factor], " + result_name + "[factor]);\n";
                glue_string += "}\n";
            }
            else
            {
                glue_string += "atomicAdd(&y_arr[" + row_name + " * K + dense_matrix_ptr], " + result_name + "[0]);\n";
            }
        }

        shared_ptr<glue_code_token> glue_token(new glue_code_token(input_of_glue_code, output_of_glue_code, glue_string));
        return glue_token;
    }
    else
    {
        return NULL;
    }
}