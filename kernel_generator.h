#ifndef KERNEL_GENERATOR_H
#define KERNEL_GENERATOR_H

// 一个新的头文件用来处理代码生成的三个阶段，循环阶段，reduce拼接阶段，然后是依赖分析阶段，根据分析出来的一来读取元数据
// 最后一个阶段是model driven的压缩
#include "code_source_data.hpp"
#include <map>
#include "config.hpp"
#include <unistd.h>
#include "metadata_set.hpp"
#include <string>

using namespace std;

// token类型的枚举
enum TOKEN_TYPE
{
    NONE_TOKEN_TYPE = -99,
    FOR_TOKEN_TYPE,
    // 一种简化的表达，包含了一些数学表达式的计算，对于AlphaSparse中不关注的优化部分
    MATH_EXPR_TOKEN_TYPE,
    VAR_NAME_TOKEN_TYPE,
    // MEM访问
    ARR_ACCESS_TOKEN_TYPE,
    // 数据类型，包含指针类型和
    DATA_TYPE_TOKEN_TYPE,
    // 数据的声明，包含了数据类型，变量名称以及初始化表达式
    VAR_INIT_TOKEN_TYPE,
    // 共享内存数组的声明
    SHARED_MEM_INIT_TOKEN_TYPE,
    // 共享内存内的赋值操作
    SHARED_MEM_WRITE_TOKEN_TYPE,
    // metadata get的占位符
    METADATA_GET_BASIC_TOKEN_TYPE,
    // 归约代码块的占位符
    REDUCTION_BASIC_TOKEN_TYPE,
    // for循环的占位符，包含了遍历的层次
    FOR_BASIC_TOKEN_TYPE,
    // 使用SHARED_MEM来广播一组数据
    SHARED_MEM_BROADCAST_TOKEN_TYPE,
    // 用算式或者变量给另外一个变量赋值
    VAR_ASSIGN_TOKEN_TYPE,
    // 真正的METADATA_NAME
    METADATA_GET_TOKEN_TYPE,
    // 一行CUDA代码
    RAW_CUDA_CODE_LINE,
    // 基本的胶水代码basic_glue_code
    BASIC_GLUE_CODE,
    IF_ELSE_TOKEN_TYPE,
    ARR_DECLARATION_TOKEN_TYPE
};

// 打印当前的token类型
string convert_token_type_to_string(TOKEN_TYPE token_type);
// 检查当前的token是不是正确的，防止整型的错误强制类型转换
bool check_token_type(TOKEN_TYPE token_type);

// 查看一个变量所处的位置，或者立即数
enum VAR_TYPE
{
    NONE_VAR_TYPE = -9,
    GLOBAL_MEM_VAR_TYPE,
    SHARED_MEM_VAR_TYPE,
    REGISTER_VAR_TYPE,
    CONSTANT_VAR_TYPE,
};

// 打印当前变量类型
string convert_var_type_to_string(VAR_TYPE var_type);
// 检查变量类型是不是正确
bool check_var_type(VAR_TYPE var_type);

enum REDUCTION_TOKEN_TYPE
{
    NONE_REDUCTION_TOKEN_TYPE = -80,
};

string convert_reduction_token_type_to_string(REDUCTION_TOKEN_TYPE reduction_token_type);
bool check_reduction_token_type(REDUCTION_TOKEN_TYPE reduction_token_type);

// 胶水的类型
enum GLUE_CODE_TOKEN_TYPE
{
    NONE_GLUE_CODE_TOKEN_TYPE =  -70,
};

string convert_glue_code_token_type_to_string(GLUE_CODE_TOKEN_TYPE glue_code_token_type);
bool check_glue_code_token_type(GLUE_CODE_TOKEN_TYPE glue_code_token_type);

// 语法树的基本单元，token，包含名字、类型、以及子树
class basic_token
{
public:
    // 执行
    basic_token(bool is_terminal, TOKEN_TYPE type);

    // 写子节点的token，包含了其作为儿子节点的名字
    void set_token_of_child(string child_token_name, shared_ptr<basic_token> token);

    // token的名字，对应到变量名、函数名
    shared_ptr<basic_token> get_token_of_child(string child_token_name);

    // 查看某个子节点是不是存在
    bool child_is_exist(string child_token_name);

    // 删除某个子节点
    void remove_child(string child_token_name);

    virtual string get_inited_var_name()
    {

    }

    // 获得当前节点的名字
    // string get_var_name()
    // {
    //     return var_name;
    // }

    // 查看当前是不是终结符
    bool get_is_terminal()
    {
        return is_terminal;
    }

    // 获取当前的token类型
    TOKEN_TYPE get_token_type()
    {
        return token_type;
    }

    // 执行当前的子树，用当前根节点生成代码，需要被重载的函数，反之生成一个注释
    virtual string run()
    {
        return "// basic token without any implementation\n";
    }

    // 执行静态检查的代码
    virtual bool static_check() = 0;

protected:
    map<string, shared_ptr<basic_token>> token_of_child_map;
    // 当前token的名字。token有两个名字，一个是其真正的名字，用来代码生成，一个是其作为儿子节点的名字，用来从父节点索引
    // string var_name;
    // 查看是不是终结符
    bool is_terminal = false;

    TOKEN_TYPE token_type = NONE_TOKEN_TYPE;
};


// 变量名的token
class var_name_token : public basic_token
{
public:
    var_name_token(string var_str, VAR_TYPE var_type);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();

    VAR_TYPE get_var_type()
    {
        return this->var_type;
    }

    // 获得名字
    string get_var_name_str()
    {
        return this->var_str;
    }

private:
    // 变量名的类型
    VAR_TYPE var_type = NONE_VAR_TYPE;
    // 变量字符串，是变量名或者常量
    string var_str = "";
};

class arr_access_token : public basic_token
{
public:
    arr_access_token(shared_ptr<var_name_token> dest_var_name, shared_ptr<var_name_token> mem_ptr_name, shared_ptr<var_name_token> mem_index);

    // 访存的索引实际上可以是一个小表达式
    arr_access_token(shared_ptr<var_name_token> dest_var_name, shared_ptr<var_name_token> mem_ptr_name, string mem_index);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();
private:
    string mem_index;
};

// 数学计算的表达式，包含了所有的算式，内部不再设立子节点。这里是纯计算的，不能出现memory access
// 是终结符
class math_expr_token : public basic_token
{
public:
    math_expr_token(string math_expression_str);
    math_expr_token(string math_expression_str, shared_ptr<arr_access_token> arr_acc_expr, string op);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();

private:
    string math_expression_str = "";
    string op = "";
};



// 数组访问的token，单独提取出来主要是为了执行model driven优化
// 包含三个子节点：mem_ptr_name, mem_index，dest_var_name。三个都是VAR_NAME_TOKEN
// mem_ptr_name是GLOBAL或者SHARED类型，mem_index是REGISTER或者CONSTANT类型
// dest_var_name是REGISTER类型
// TODO：对于GLOBAL的变量来说，要将当前的mem_ptr_name和一个metadata item绑定起来


// 数据类型，包含具体的数据类型以及其是否是指针的标识
class data_type_token : public basic_token
{
public:
    data_type_token(data_type type, bool is_pointer);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();

    // 获得数据类型
    data_type get_data_type()
    {
        return this->type;
    }

    // 查看是不是指针
    bool get_is_pointer()
    {
        return this->is_pointer;
    }

private:
    data_type type = NONE_DATA_TYPE;
    bool is_pointer;
};

// 初始化，包含三个内容，一个是数据类型，一个是变量名，一个是初始化表达式
// 其中初始化表达式的可选的
// 数据类型的key名是data_type_declare
// 变量名的key名是init_var_name，是REGISTER的VAR_NAME类型
// 初始化表达式为init_math_express
class var_init_token : public basic_token
{
public:
    var_init_token(shared_ptr<data_type_token> data_type_declare, shared_ptr<var_name_token> init_var_name, shared_ptr<math_expr_token> init_math_express);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();

    // 获得var_name
    string get_inited_var_name();
};

// 共享内存的初始化，包含了一个必然不是指针的数据类型，一个shared mem的变量名，一个用来声明shared memory大小的变量名
// 数据类型的key名为data_type_declare
// shared mem变量的key名为init_shared_mem_var_name，是SHARED的VAR_NAME类型
// shared mem大小的变量名为shared_mem_size_var_name，是RESITER或者CONSTANT类型
class shared_mem_init_token : public basic_token
{
public:
    shared_mem_init_token(shared_ptr<data_type_token> data_type_declare, shared_ptr<var_name_token> init_shared_mem_var_name, shared_ptr<var_name_token> shared_mem_size_var_name);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();
};

// 将内容写到对应的shared_memory，包含三个变量，第一个是SHARED类型，第二个和第三个是REGISTER或者CONSTANT类型
// VAR_NAME1[VAR_NAME2]=VAR_NAME3
class shared_mem_write_token : public basic_token
{
public:
    shared_mem_write_token(shared_ptr<var_name_token> shared_mem_name, shared_ptr<var_name_token> input_index, shared_ptr<var_name_token> written_value);

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();
};

// metadata get
class metadata_get_basic_token : public basic_token
{
public:
    // 产生一个默认构造函数，就是空的
    metadata_get_basic_token(POS_TYPE token_position);

    POS_TYPE get_token_position()
    {
        return token_position;
    }

    // 实现执行函数
    string run();

    // 实现静态检查函数
    bool static_check();

private:
    // 当前获取元数据的层次
    POS_TYPE token_position = NONE_META;
};

// 一个reduction的输入或者输出，一般内部是一个或者多个变量（在派生类中实现）。变量有其对应的类型，其中变量是一个token的形式，可以直接运行来输出
class basic_IO_of_reduction
{
public:
    basic_IO_of_reduction(string name_of_IO_type, POS_TYPE generated_pos)
    {
        this->name_of_IO_type = name_of_IO_type;
        this->generated_pos = generated_pos;
    }

    // 获得IO类型的名字
    string get_name_of_IO_type()
    {
        return name_of_IO_type;
    }

    // 当前这个IO变量是哪一个层级产生的
    POS_TYPE get_generated_pos()
    {
        return generated_pos;
    }

    virtual unsigned int get_count()
    {

    }

    virtual shared_ptr<var_name_token> var_name_token_of_IO_register()
    {

    }

    virtual vector<shared_ptr<var_name_token>> var_names_token_of_IO_register()
    {

    }

    // 将当前的IO用string输出，用以打印log
    virtual string to_string()
    {
        string return_str = "basic_IO_of_reduction::{\nname_of_IO_type:" + name_of_IO_type + ",\n generated_pos:" + convert_pos_type_to_string(generated_pos) + ",\n";
        
        // 将非绑定变量打印出来
        for (int i = 0; i < unbounded_var_name_vec.size(); i++)
        {
            return_str = return_str + "unbounded_var" + ":" + unbounded_var_name_vec[i]->run() + ",\n";
        }
        
        return_str = return_str + "\n}";
        return return_str;
    }

    // 绑定变量的数量
    int number_of_unbound_var_num()
    {
        return unbounded_var_name_vec.size();
    }

    // 读出一个绑定变量出来
    shared_ptr<var_name_token> get_unbound_var(int var_index)
    {
        assert(var_index >= 0);
        return unbounded_var_name_vec[var_index];
    }

    // 加入一个绑定变量
    void add_var_name(shared_ptr<var_name_token> input_var_name)
    {
        assert(input_var_name != NULL);
        // 不能有重复的变量
        for (int i = 0; i < this->unbounded_var_name_vec.size(); i++)
        {
            assert(input_var_name->run() != this->unbounded_var_name_vec[i]->run());
        }

        this->unbounded_var_name_vec.push_back(input_var_name);
    }

private:
    string name_of_IO_type = "basic_IO_of_reduction";
    POS_TYPE generated_pos = NONE_META;

protected:
    // 用一个数组来记录输入和输出中所包含的变量名，会和外界交互的非绑定变量名
    vector<shared_ptr<var_name_token>> unbounded_var_name_vec;
};

// 裸的CUDA代码，而且是一行代码
class raw_cuda_code_line : public basic_token
{
public:
    raw_cuda_code_line(string init_str)
        : basic_token(true, RAW_CUDA_CODE_LINE)
    {
        code_line = init_str;
    }

    string run()
    {
        return code_line;
    }

    void append(string append_str)
    {
        code_line = code_line + append_str;
    }

    bool static_check()
    {
        return true;
    }

private:
    string code_line = "";
};

// 用户高度自定义的reduction_basic_token需要code generator来申请资源，所以涉及循环引用
class code_generator;

class reduction_basic_token : public basic_token
{
public:
    reduction_basic_token(POS_TYPE token_position, shared_ptr<meta_data_set> meta_data_set_ptr = NULL, shared_ptr<code_generator> code_generator_ptr = NULL);

    POS_TYPE get_token_position()
    {
        return this->token_position;
    }

    virtual shared_ptr<basic_IO_of_reduction> get_output_IO()
    {
    }


    virtual shared_ptr<basic_IO_of_reduction> get_input_IO()
    {
    }
    // 插入一个AlphaSparse类别的语元，要求是一个codeline
    // 在chile map中数据分别使用AlphaSparse_code和raw_cuda_code标记，并且根据其插入的顺序标记上123，在run中一一输出就好了
    // 在插入的过程分别搜索两种code在同一个尾部标记是否存在，应该是只有一个是存在的，那么打印那一个就好了。其他情况都代表错误。这和metadata set是截然不同的
    void add_alpha_code_line(shared_ptr<basic_token> alpha_code_line);

    void add_raw_cuda_code_line(shared_ptr<raw_cuda_code_line> raw_cuda_code_line);

    // 实现执行函数
    virtual string run();

    // 实现静态检查函数
    bool static_check();

protected:
    POS_TYPE token_position = NONE_META;

    // 引入一个代码生成器，用以生成代码。这里使用weak智能指针的方式
    // 在run函数中这个指针会被复原成一般的智能指针，而这个智能指针是一个局部变量而不是数据成员
    // 从而规避了循环引用。在run函数执行完之后，对应指针会被析构
    weak_ptr<code_generator> code_generator_ptr;

    // 包含一个元数据库，是生成代码过程中必须的
    shared_ptr<meta_data_set> meta_data_set_ptr;
};

// 胶水代码的基类，里面说明要处理的那种类型reduction IO
class basic_glue_code : public basic_token
{
public:
    // 初始化，使用输入和输出两组变量来进行初始化，也就是对应的IO
    basic_glue_code(shared_ptr<basic_IO_of_reduction> input_IO, shared_ptr<basic_IO_of_reduction> output_IO);

    // 执行对应的胶水代码
    virtual string run();

    // 静态检查
    bool static_check();

private:
    vector<shared_ptr<var_name_token>> input_var_vec;
    vector<shared_ptr<var_name_token>> output_var_vec;
};


class glue_code_token : public basic_glue_code
{
public:
    glue_code_token(shared_ptr<basic_IO_of_reduction> input_IO, shared_ptr<basic_IO_of_reduction> output_IO, string run_result);

    // 执行对应的胶水代码
    virtual string run();

    // 静态检查
    bool static_check();
private:
    string run_result;
};



// 其内部包含三个子结构，metadata_get_basic_token，reduction_basic_token，内部嵌套循环的for_basic_token
// reduction_basic_token未必存在，inner_loop未必存在，但是metadata_get_basic_token必然存在
// 不存在的就传入空指针
// 在map中的key名和数据成员的名字是一致的。
class for_basic_token : public basic_token
{
public:
    for_basic_token(POS_TYPE token_position, shared_ptr<metadata_get_basic_token> metadata_get_code, shared_ptr<for_basic_token> inner_loop, shared_ptr<reduction_basic_token> reduction_code, shared_ptr<basic_glue_code> glue_code_block);

    virtual string run();

    // 静态检查
    bool static_check();

    POS_TYPE get_token_position()
    {
        return this->token_position;
    }

    // 查看子for循环的token
    virtual shared_ptr<for_basic_token> get_child_for_token();

    // 查看循环中的reduction
    virtual shared_ptr<reduction_basic_token> get_reduction_token();

    // 获得循环中的元数据获取
    virtual shared_ptr<metadata_get_basic_token> get_metadata_get_token();

    // 使用一个函数，插入三个内容的指针：元数据获取，归约和胶水
    virtual void set_metadata_get_code(shared_ptr<metadata_get_basic_token> token_ptr);

    virtual void set_reduction_code(shared_ptr<reduction_basic_token> token_ptr);

    virtual void set_glue_code_block(shared_ptr<basic_glue_code> token_ptr);

protected:
    // 将代码的生成分为函数头和函数体两个部分
    virtual string for_header_run();

    virtual string for_body_run();

private:
    POS_TYPE token_position = NONE_META;
};

// for循环，一共有多个成员，一个是哨兵变量，一个是哨兵变量的起始位置，一个是哨兵变量的结束位置（半开区间），一个是哨兵变量前进的步长
// 哨兵变量的类型loop_var_name_type, DATA_TYPE_TOKEN，必然不是指针类型
// 哨兵变量loop_var_name，VAR_NAME_TOKEN，必然是REGISTER类型
// 哨兵变量的起始位置，begin_loop_var_name，VAR_NAME_TOKEN，REGISTER或者CONSTANT类型
// 哨兵变量的结束位置，end_loop_var_name，VAR_NAME_TOKEN，REGISTER或者CONSTANT类型
// 哨兵变量的自增，step_loop_var_name，VAR_NAME_TOKEN，REGISTER或者CONSTANT类型
// 内部有三个子分量，一个是metadata get，一个是inner loop，一个是reduction，这三个分量可以是，只输出注释，之后再一点点填充
// 而内部的三个分量，由父类的中的内容来处理了。
class for_token : public for_basic_token
{
public:
    // 循环的起始，结束和前进的步长三个token可以是变量也可以是常量
    for_token(shared_ptr<data_type_token> loop_var_name_type, shared_ptr<var_name_token> loop_var_name,
              shared_ptr<var_name_token> begin_loop_var_name, shared_ptr<var_name_token> end_loop_var_name,
              shared_ptr<var_name_token> step_loop_var_name, POS_TYPE token_position, shared_ptr<metadata_get_basic_token> metadata_get_code,
              shared_ptr<for_basic_token> inner_loop, shared_ptr<reduction_basic_token> reduction_code, shared_ptr<basic_glue_code> glue_code_block);

    for_token(shared_ptr<data_type_token> loop_var_name_type, shared_ptr<var_name_token> loop_var_name,
              shared_ptr<basic_token> begin_loop_var_name, shared_ptr<basic_token> end_loop_var_name,
              shared_ptr<var_name_token> step_loop_var_name, POS_TYPE token_position, shared_ptr<metadata_get_basic_token> metadata_get_code,
              shared_ptr<for_basic_token> inner_loop, shared_ptr<reduction_basic_token> reduction_code, shared_ptr<basic_glue_code> glue_code_block);


    // 实现静态检查函数
    bool static_check();

    // 重新实现for循环的头部
    string for_header_run();
};

// 使用共享内存来广播数据，用一个数组来存储要广播的globalmemory，包含了一系列GLOBAL类型的变量。global_mem_read_arr
// 用一个数组来存储每一个globalmemory数组要读取的索引，一定是REGISTER类型。global_mem_read_index
// 用一个数组存储每一个全局内存的类型，一定是data type的token。data_type_of_read_data
// 用一个数组存读取出来存放的变量名，一定是REGISTER类型。dest_variable
// 在child_map中，在key的末尾加上“_数字”来表达数组对应的元素
// 用来广播的shared_mem空间统一使用global_mem_read_arr的名字加上“_shared_space”来命名
class shared_mem_broadcast_token : public basic_token
{
public:
    shared_mem_broadcast_token(vector<shared_ptr<data_type_token>> data_type_of_read_data_arr,
                               vector<shared_ptr<var_name_token>> global_mem_read_arr, vector<shared_ptr<var_name_token>> global_mem_read_index_arr,
                               vector<shared_ptr<var_name_token>> dest_variable_arr);

    // 还有一个默认构造函数，生成一个空的内容，之后可以一个个往里面加
    shared_mem_broadcast_token();

    // 加入要广播的数据的声明
    void add_broadcast_data(shared_ptr<data_type_token> data_type_of_read_data, shared_ptr<var_name_token> global_mem_read, shared_ptr<var_name_token> global_mem_read_index, shared_ptr<var_name_token> dest_variable);

    // 实现静态检查函数
    bool static_check();

    // 用一个函数给出广播数据所需要的共享内存名字
    vector<string> needed_shared_mem_name();

    // 用一个函数给出所有所需的共享内存的大小
    vector<unsigned int> needed_shared_mem_array_size();

    // 用一个函数给出所有的数据类型
    vector<data_type> needed_shared_mem_data_type();

    // 执行代码生成
    virtual string run();

private:
    int broadcast_data_num = 0;
};

// 变量的赋值，主要是两个变量之间画个等号，以及变量和算式之间放等号
// 等号左边的变量是REGISTER类型的，右边的可以是算式，可以是REGISTER和CONSTANT类型的变量
class var_assign_token : public basic_token
{
public:
    var_assign_token(shared_ptr<var_name_token> left_operand, shared_ptr<var_name_token> right_operand);

    var_assign_token(shared_ptr<var_name_token> left_operand, shared_ptr<math_expr_token> right_operand);

    // 实现静态检查函数
    bool static_check();

    // 执行代码生成
    string run();
};

class if_else_token : public basic_token
{
public:
    if_else_token(vector<shared_ptr<math_expr_token>> condition, vector<unsigned int> ptr, vector<shared_ptr<var_assign_token>> assign_expr);
    bool static_check();

    string run();
private:
    vector<unsigned int> ptr;
};


// 真正元数据获取，当中包含了var init，var assign，var access，shmem_bc四种类型的计算，其中init类型的表达式要放在最前面，剩下的表达式按照输入优先级
// init类型的子token都称为init_expr，剩下的都是metadata_get_expr
class metadata_set_get_token : public metadata_get_basic_token
{
public:
    metadata_set_get_token(POS_TYPE pos);

    void add_metadata_get_expr(shared_ptr<var_init_token> init_expr);

    void add_metadata_get_expr(shared_ptr<var_assign_token> metadata_get_expr);

    void add_metadata_get_expr(shared_ptr<arr_access_token> metadata_get_expr);

    void add_metadata_get_expr(shared_ptr<shared_mem_broadcast_token> metadata_get_expr);

    void add_metadata_get_expr(shared_ptr<basic_token> metadata_get_expr);

    void add_special_assign_expr(shared_ptr<basic_token> assign_expr);

    // 实现静态检查函数
    bool static_check();

    // 执行代码生成
    string run();

private:
    // 当前init类型的表达式数量
    int num_of_var_init = 0;
    int num_of_metadata_get_expr = 0;
    int num_of_special_assign_expr = 0;
};


class arr_declaration_token : public basic_token
{
public:
    arr_declaration_token(shared_ptr<data_type_token> data_type_declare, shared_ptr<var_name_token> mem_ptr_name, shared_ptr<math_expr_token> mem_size);

    // 实现执行函数
    string run();
    string get_inited_var_name();
    // 实现静态检查函数
    bool static_check();
};

// 命名规则，对于global access访问的变量的规则
// 从其他的更外层的并行层次引入的变量，命名规范根据其读取的数据名，以及索引名拼接而成，仅仅包含索引名中的字母和数据，其他的所有符号用下划线代替，最后一item一词结尾
string var_of_metadata_from_spec_paral(POS_TYPE read_pos_type, string read_metadata_name, int read_sub_matrix_id, shared_ptr<math_expr_token> index_expr);


string code_of_data_type(data_type type);


void write_string_to_file(string file_name, string output_str);


// 根据IO的类型返回对应glue code的token

#endif