#ifndef CODE_GENERATOR_H
#define CODE_GENERATOR_H

#include "metadata_set.hpp"
#include "kernel_generator.h"

using namespace std;

// 用一个数据结构来打包所有的元数据相关的注册数据
typedef struct back_up_of_metadata_register
{
    // 用一个表来存储所有需要的Metadata表项，这个表包含三个数组，分别记录一个Metadata item的POS，名字和所属的子矩阵
    // 这个表有几个用途：1）用来注明Metadata的数据依赖，2）用来注明kernel的形参，3）用来注明要通过文件系统从系统传递给内核程序的数据
    vector<POS_TYPE> pos_of_needed_metadata_vec;
    vector<string> real_name_of_needed_metadata_vec;
    vector<int> sub_matrix_id_of_needed_metadata_vec;
    // vector<bool> memory_access_fusing_op_flag_vec;
    // 使用set来判断一些是不是有重复的，将上面的一条索引转换成一个字符串
    set<string> added_metadata_dependency;

    // 被合并的变量的访问用来注明每一个层次的Metadata get中被合并范围的数据的内容
    vector<POS_TYPE> pos_of_fused_metadata_vec;
    vector<string> real_name_of_fused_metadata_vec;
    vector<int> sub_matrix_id_of_fused_metadata_vec;
    // 用一个数组来存储表达式
    vector<shared_ptr<math_expr_token>> access_index_fused_metadata_vec;

    // 当前全局变量的申请代码
    vector<shared_ptr<basic_token>> global_var_init_token_vec;

    // 共享内存的相关注册
    vector<data_type> needed_shared_mem_data_type_vec;
    vector<string> needed_shared_mem_name_vec;
    vector<unsigned int> needed_shared_mem_size_vec;

}back_up_of_metadata_register_t;




// 代码生成器，一些代码生成的元数据，同时也是代码生成语法树的根节点，包含了整个程序的基本结构
// 为了简洁的实现，不采用自动化的AST优化，相关Operator需要把对应层次的访存合并相关的写好
// 一个code_generator只能生成一个内核，出现多流的情况需要多个code_generator的实例
class code_generator
{
public:
    // 使用元数据库的指针来初始化
    code_generator(shared_ptr<meta_data_set> meta_data_set_ptr, int sub_matrix_id)
    {
        assert(meta_data_set_ptr != NULL);
        assert(sub_matrix_id >= 0);
        this->meta_data_set_ptr = meta_data_set_ptr;
        this->sub_matrix_id = sub_matrix_id;
        this->BMTB_id = make_shared<var_name_token>("BMTB_id", REGISTER_VAR_TYPE);
        this->BMW_id = make_shared<var_name_token>("BMW_id", REGISTER_VAR_TYPE);
        this->BMT_id = make_shared<var_name_token>("BMT_id", REGISTER_VAR_TYPE);
    }

    // 获得当前代码生成器所属于的子矩阵
    int get_sub_matrix_id()
    {
        return this->sub_matrix_id;
    }

    // 加入一个数据依赖
    void add_new_metadata_dependency(POS_TYPE pos, string real_name, int sub_matrix_id);

    // 加入一个被合并的元数据访问
    void add_new_fused_metadata_access(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<math_expr_token> metadata_access_index_expr);

    // 做一个检查，查看相关元数据是不是匹配
    bool check();

    // 查看是不是存在一个数据依赖
    bool if_dependency_exist(POS_TYPE pos, string real_name, int sub_matrix_id);

    // 查看一个被合并的访问是不是存在
    bool if_fused_metadata_access_exist(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<math_expr_token> metadata_access_index_expr);

    bool if_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id);
    bool if_branch_compress(POS_TYPE pos, string real_name, int sub_matrix_id);
    bool if_cycle_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id);
    bool if_cycle_increase_compress(POS_TYPE pos, string real_name, int sub_matrix_id);
    bool if_residual_compress(POS_TYPE pos, string real_name, int sub_matrix_id);

    shared_ptr<basic_token> get_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr);
    shared_ptr<basic_token> get_branch_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr);
    shared_ptr<basic_token> get_cycle_linear_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr);
    shared_ptr<basic_token> get_cycle_increase_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr);
    shared_ptr<basic_token> get_residual_compress(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr);
    shared_ptr<basic_token> get_compress_and_relative_result(POS_TYPE pos, string real_name, int sub_matrix_id, shared_ptr<var_name_token> var_token, shared_ptr<math_expr_token> access_index_expr);

    // 生成代码，先生成文件IO的代码
    string generate_matrix_format_read_code();

    // 下面的几个函数需要一个参数，即遍历内核的遍历次数
    // 生成内核调用和周边的性能测试代码
    string generate_profiling_code_and_kernel(unsigned int kernel_repeat_number);

    // 生成内核调用代码，一般就一行，根据配置来选择运行几次
    // 吞吐量和延迟有不同的测试方法
    string generate_kernel_calling_code(unsigned int kernel_repeat_number);

    // 生成main函数
    string generate_main_function_code(unsigned int kernel_repeat_number);

    // 整个代码程序，只考虑一个子矩阵，在op graph中执行器中有支持多流以及多个子矩阵的版本
    // 文件的生成位置和元数据在同一个文件夹内
    void generate_kernel_file(unsigned int kernel_repeat_number);

    // 按照已经注明的数据来向硬盘中存储，返回存储的ID，TODO这里面可能存在数组合并的优化
    unsigned long write_matrix_format_to_disk();

    // 设置线程块和线程
    void set_thread_grid(vector<unsigned int> grid, vector<unsigned int> block);

    vector<unsigned int> get_grid()
    {
        return this->thread_block_num;
    }

    vector<unsigned int> get_block()
    {
        return this->thread_num;
    }

    void generate_loop_var(bool thread_x_direction_reuse);

    // 一个kernel函数的函数头
    string generate_header_of_kernel_declaration_code();

    // 一个kernel的声明
    string generate_kernel_declaration_code();

    // 获得线程数量的代码
    string total_thread_num_code();

    // 获得线程块数量代码
    string total_thread_block_num_code();

    // 获得warp总数量
    string total_warp_num_code();

    // 获得每一个线程块中线程的数量
    string thread_num_in_thread_block_code();

    // 获得每一个线程块中warp的数量
    string warp_num_in_thread_block_code();

    // 获得线程块的全局ID
    string global_thread_block_id_code();

    // 获得线程的全局ID
    string global_thread_id_code();

    // 获得warp的全局ID
    string global_warp_id_code();

    // 获得线程块内的warp索引
    string warp_id_in_thread_block_code();

    // 获得线程块内的线程索引
    string thread_id_in_thread_block_code();

    // 获得向量宽度
    string vector_width_code();

    // 获得线程的warp内ID
    string thread_id_in_warp_code();

    // 根据当前需要生成线程网格相关的代码，是在函数头部的代码
    string generate_code_of_grid_info_calculate();


    // 根据位置生成对应数据块的索引变量
    shared_ptr<var_name_token> code_of_data_block_id_distributed_to_spec_paral(POS_TYPE pos);

    shared_ptr<metadata_set_get_token> generate_token_of_fused_metadata_get_in_spec_paral_level(POS_TYPE pos);

    // 经过memory access fusion优化的数组访问，返回一个变量，访问索引的表达式本质上也可以是一个变量
    shared_ptr<var_name_token> generate_fused_memory_access(POS_TYPE pos, string metadata_name, shared_ptr<math_expr_token> mem_access_index_expr);

    shared_ptr<var_name_token> generate_fused_memory_access_with_relative(POS_TYPE pos, string metadata_name, shared_ptr<math_expr_token> mem_access_index_expr);

    // 没有合并的数组访问，返回两个token，一个是变量声明，一个是metadata数组的访问（也可以是一个赋值表达式），访问索引的表达式本质上也可以是一个变量
    // 也就是一个输出是一个VAR_INIT_TOKEN_TYPE，一个是ARR_ACCESS_TOKEN_TYPE
    vector<shared_ptr<basic_token>> generate_unfused_memory_access(POS_TYPE pos, string metadata_name, shared_ptr<math_expr_token> mem_access_index_expr, bool flag, string row_name = "");

    // 根据不同层次并行的需要，给出for循环的结构，从thread层次开始，从内向外构造，用一个变量来判断是不是最外层的循环
    // 因为最外层循环的直接取决于数据块的个数而不是父块的偏移量
    shared_ptr<for_token> generate_for_token_of_spec_paral_level(POS_TYPE pos, shared_ptr<for_token> child_for_token, bool the_outest_loop);

    // 打开特定级别的并行，在代码生成的时候
    void open_spec_level_of_paral(POS_TYPE pos);

    // 生成代码，根据线程网格的记录子并行级别执行粒度在父并行级别的一个粒度中的数量
    shared_ptr<var_name_token> code_of_thread_grid_info_in_spec_paral(POS_TYPE child_pos, POS_TYPE parent_pos);

    // 查看之前有没有声明过相同的共享内存
    bool shared_mem_is_exist(string shared_mem_name);

    // 增加一个新的共享内存
    void add_new_use_of_shared_mem(data_type shared_mem_data_type, string shared_mem_name, unsigned int shared_mem_size);

    // 一个函数，生成共享内存数组声明的代码
    string generate_code_of_shared_memory_array_declaration();

    // 将交错存储置为true
    void set_interleave_storage();

    // 查看当前是不是使用交错存储
    bool get_interleave_storage();

    // 添加一个全局变量
    shared_ptr<var_name_token> generate_global_var(data_type type, string var_name, shared_ptr<math_expr_token> var_init_expr_token, unsigned int size = 0);

    // 查看全局变量是不是存在
    bool global_var_is_existing(string var_name);

    // 生成所有的全局变量的声明
    string generate_global_var_init_code();

    // 根据开启的并行层次组合出对应的嵌套的for token，也就是stage1
    shared_ptr<for_token> generate_for_structure_of_kernel();

    // 生成嵌套循环结构，并且设置root for
    void generate_for_structure_of_kernel_and_set_root_for_token();

    // 将所有归约方法插入到for循环的结构中
    void insert_reduction_token_to_for_structure();

    // 将所有元数据获取插入到for循环的结构中
    void insert_metadata_get_token_to_for_structure();

    // 将归约方法插入到for循环的特定并行层次
    void insert_reduction_token_to_spec_paral_of_for_structure(POS_TYPE pos);

    // 将元数据收集的代码token放到循环的对应位置
    void insert_metadata_get_token_to_spec_paral_of_for_structure(POS_TYPE pos);

    bool reduction_token_is_existing(POS_TYPE pos);

    // 设定特定级别的归约token
    void set_reduction_token(POS_TYPE pos, shared_ptr<reduction_basic_token> token_ptr);

    void generate_glue_code();

    shared_ptr<basic_glue_code> get_glue_code_token_according_to_input_and_ouput_IO(shared_ptr<basic_IO_of_reduction> input_of_glue_code, shared_ptr<basic_IO_of_reduction> output_of_glue_code);

    string code_of_root_for_token();

    string code_of_for_structure();

    back_up_of_metadata_register_t backup_metatdata_and_global_var();

    void recover_metatdata_and_global_var(back_up_of_metadata_register_t back_up);

    void compile_and_set_for_code();

    void set_thread_for_row(bool flag)
    {
        this->thread_for_row = flag;
    }

    void compile()
    {
        assert(this->root_for_token_ptr == NULL || this->compiled_for_code == "");
        this->compile_and_set_for_code();
    }

    unsigned long generate_final_program(int repeat_num = 10)
    {
        // 创建format文件
        unsigned long output_id = this->write_matrix_format_to_disk();

        // 创建代码文件
        this->generate_kernel_file(repeat_num);

        return output_id;
    }

    void set_dense_array(shared_ptr<universal_array> dnese_array)
    {
        this->dense_array = dnese_array;
    }

    shared_ptr<meta_data_set> get_metadata_set()
    {
        return this->meta_data_set_ptr;
    }

    void set_for_loop_begin_ptr(POS_TYPE pos, shared_ptr<var_name_token> begin_ptr)
    {
        if (pos == TBLOCK_META)
        {
            this->BMTB_begin_ptr = begin_ptr;
        }
        else if (pos == WARP_META)
        {
            this->BMW_begin_ptr = begin_ptr;
        }
        else if (pos == THREAD_META)
        {
            this->BMT_begin_ptr = begin_ptr;
        }
        else
        {
            assert(false);
        }
    }

    shared_ptr<var_name_token> get_for_loop_begin_ptr(POS_TYPE pos)
    {
        if (pos == TBLOCK_META)
        {
            return this->BMTB_begin_ptr;
        }
        else if (pos == WARP_META)
        {
            return this->BMW_begin_ptr;
        }
        else if (pos == THREAD_META)
        {
            return this->BMT_begin_ptr;
        }
        else
        {
            assert(false);
        }
    }

    void set_for_loop_step(POS_TYPE pos, shared_ptr<var_name_token> step)
    {
        if (pos == TBLOCK_META)
        {
            this->BMTB_step = step;
        }
        else if (pos == WARP_META)
        {
            this->BMW_step = step;
        }
        else if (pos == THREAD_META)
        {
            this->BMT_step = step;
        }
        else
        {
            assert(false);
        }
    }

private:
    // 用一个表来存储所有需要的Metadata表项，这个表包含三个数组，分别记录一个Metadata item的POS，名字和所属的子矩阵
    // 这个表有几个用途：1）用来注明Metadata的数据依赖，2）用来注明kernel的形参，3）用来注明要通过文件系统从系统传递给内核程序的数据
    vector<POS_TYPE> pos_of_needed_metadata_vec;
    vector<string> real_name_of_needed_metadata_vec;
    vector<int> sub_matrix_id_of_needed_metadata_vec;
    // vector<bool> memory_access_fusing_op_flag_vec;
    // 使用set来判断一些是不是有重复的，将上面的一条索引转换成一个字符串
    set<string> added_metadata_dependency;

    // 被合并的变量的访问用来注明每一个层次的Metadata get中被合并范围的数据的内容
    vector<POS_TYPE> pos_of_fused_metadata_vec;
    vector<string> real_name_of_fused_metadata_vec;
    vector<int> sub_matrix_id_of_fused_metadata_vec;
    // 用一个数组来存储表达式
    vector<shared_ptr<math_expr_token>> access_index_fused_metadata_vec;

    // 当前全局变量的申请代码
    vector<shared_ptr<basic_token>> global_var_init_token_vec;

    // 当前的代码生成器所关注的子矩阵
    int sub_matrix_id = -999;

    // 用一个元数据库来分析数据依赖，查看具体的数据是不是真的是在具体的Metadata set中存在
    shared_ptr<meta_data_set> meta_data_set_ptr = NULL;

    // 查看所需要的并行层级
    bool need_tblock_level_paral = false;
    bool need_warp_level_paral = false;
    bool need_thread_level_paral = false;

    // 设定输出的ID
    unsigned long output_id = 0;

    // 用一个布尔查看是不是存在交错存储
    bool is_interleave_storaged = false;
    bool thread_for_row = false;


    vector<unsigned int> thread_block_num;
    vector<unsigned int> thread_num;

    // 头部的一些代码，获取网格的结构，查看需要的代码
    bool need_the_whole_thread_block_number = false;
    bool need_the_whole_warp_num = false;
    bool need_the_whole_thread_num = false;
    bool need_warp_number_in_thread_block = false;
    bool need_thread_number_in_thread_block = false;

    // 头部的一些代码，当前内核函数所运行在的全局线程号，warp号，线程块号
    bool need_global_thread_id = false;
    bool need_global_thread_block_id = false;
    bool need_global_warp_id = false;

    // 相对于线程块的thread和warp的号码
    bool need_thread_id_in_thread_block = false;
    bool need_warp_id_in_thread_block = false;

    // 相对于warp的线程ID
    bool need_thread_id_in_warp = false;

    // 用三个现成的指针分别存储映射到不同并行级别的数据块的编号
    shared_ptr<var_name_token> BMTB_id = NULL;
    shared_ptr<var_name_token> BMW_id = NULL;
    shared_ptr<var_name_token> BMT_id = NULL;

    shared_ptr<var_name_token> BMTB_begin_ptr = NULL;
    shared_ptr<var_name_token> BMW_begin_ptr = NULL;
    shared_ptr<var_name_token> BMT_begin_ptr = NULL;

    shared_ptr<var_name_token> BMTB_step = NULL;
    shared_ptr<var_name_token> BMW_step = NULL;
    shared_ptr<var_name_token> BMT_step = NULL;


    // 在某一个层次是否使用非零元相对索引，分别是warp级别和线程级别
    bool nz_relative_in_thread_level = false;
    bool nz_relative_in_warp_level = false;

    // 在某一个层次是否使用行相对索引，只有warp级别和线程级别
    bool row_relative_in_thread_level = false;
    bool row_relative_in_warp_level = false;

    // 用一个数组来存储所有的共享内存的数组，包含了数组的大小和名字
    vector<data_type> needed_shared_mem_data_type_vec;
    vector<string> needed_shared_mem_name_vec;
    vector<unsigned int> needed_shared_mem_size_vec;

    shared_ptr<universal_array> dense_array = NULL;
    // 用一个指针存储多个层次的归约
    shared_ptr<reduction_basic_token> thread_level_reduction_token_ptr = NULL;
    shared_ptr<reduction_basic_token> warp_level_reduction_token_ptr = NULL;
    shared_ptr<reduction_basic_token> thread_block_level_reduction_token_ptr = NULL;
    shared_ptr<reduction_basic_token> global_level_reduction_token_ptr = NULL;

    // 使用三个指针处理胶水的code
    shared_ptr<basic_glue_code> glue_code_in_thread_level_ptr = NULL;
    shared_ptr<basic_glue_code> glue_code_in_warp_level_ptr = NULL;
    shared_ptr<basic_glue_code> glue_code_in_thread_block_level_ptr = NULL;

    // 用一个指针来存储根部的for循环结构
    shared_ptr<for_token> root_for_token_ptr = NULL;

    // for循环的代码和global reduction代码
    string compiled_for_code = "";
};

#endif