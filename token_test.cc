#include <iostream>
#include <memory>
#include "code_source_data.hpp"
#include <map>
#include "metadata_set.hpp"
#include "data_transform_step.hpp"
#include "operator.hpp"
#include "kernel_generator.h"
#include "term_print.hpp"
#include "code_generator.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include "operator_executer.hpp"
#include "reduction_token.hpp"
#include "executor.hpp"

using namespace std;

void test_math_expr_token()
{
    print_red_str_to_term("test_math_expr_token");
    shared_ptr<basic_token> math_expr_token_ptr(new math_expr_token("ab+c"));

    cout << math_expr_token_ptr->run();

    assert(math_expr_token_ptr->static_check() == true);

    shared_ptr<basic_token> math_expr_token_ptr2(new math_expr_token("ab[c]"));

    assert(math_expr_token_ptr2->static_check() == false);
}

void test_var_name_token()
{
    print_red_str_to_term("test_var_name_token");
    shared_ptr<basic_token> var_name_token_ptr(new var_name_token("1.1", CONSTANT_VAR_TYPE));

    cout << var_name_token_ptr->run() << endl;

    assert(var_name_token_ptr->static_check() == true);

    shared_ptr<basic_token> var_name_token_ptr2(new var_name_token(".1", CONSTANT_VAR_TYPE));

    assert(var_name_token_ptr2->static_check() == false);

    shared_ptr<basic_token> var_name_token_ptr3(new var_name_token(".1", GLOBAL_MEM_VAR_TYPE));

    assert(var_name_token_ptr3->static_check() == false);

    shared_ptr<basic_token> var_name_token_ptr4(new var_name_token("asd_a", GLOBAL_MEM_VAR_TYPE));

    cout << var_name_token_ptr4->run() << endl;

    assert(var_name_token_ptr4->static_check() == true);

    shared_ptr<basic_token> var_name_token_ptr5(new var_name_token("1as", GLOBAL_MEM_VAR_TYPE));

    assert(var_name_token_ptr5->static_check() == false);
}

void test_arr_access_token()
{
    print_red_str_to_term("test_arr_access_token");
    shared_ptr<var_name_token> dest_var_token_ptr(new var_name_token("dest_var", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> dest_var_token_ptr2(new var_name_token("dest_var", GLOBAL_MEM_VAR_TYPE));

    shared_ptr<var_name_token> mem_ptr_name_token_ptr(new var_name_token("mem_ptr_name", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> mem_ptr_name_token_ptr2(new var_name_token("mem_ptr_name", GLOBAL_MEM_VAR_TYPE));

    shared_ptr<var_name_token> mem_index_token_ptr(new var_name_token("mem_index", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> mem_index_token_ptr2(new var_name_token("mem_index", GLOBAL_MEM_VAR_TYPE));

    // 创造一个新的global mem赋值的语句
    shared_ptr<arr_access_token> arr_access_token_ptr(new arr_access_token(dest_var_token_ptr, mem_ptr_name_token_ptr2, mem_index_token_ptr));
    cout << arr_access_token_ptr->run() << endl;
    assert(arr_access_token_ptr->static_check() == true);

    shared_ptr<arr_access_token> arr_access_token_ptr2(new arr_access_token(dest_var_token_ptr2, mem_ptr_name_token_ptr2, mem_index_token_ptr));
    assert(arr_access_token_ptr2->static_check() == false);

    shared_ptr<arr_access_token> arr_access_token_ptr3(new arr_access_token(dest_var_token_ptr, mem_ptr_name_token_ptr, mem_index_token_ptr));
    assert(arr_access_token_ptr3->static_check() == false);

    shared_ptr<arr_access_token> arr_access_token_ptr4(new arr_access_token(dest_var_token_ptr, mem_ptr_name_token_ptr2, mem_index_token_ptr2));
    assert(arr_access_token_ptr4->static_check() == false);

    shared_ptr<math_expr_token> mem_index_token_ptr3(new math_expr_token("a+b"));
    // 创造一个新的mem access表达式，使用数学表达式当索引
    shared_ptr<arr_access_token> arr_access_token_ptr5(new arr_access_token(dest_var_token_ptr, mem_ptr_name_token_ptr2, mem_index_token_ptr3->run()));
    assert(arr_access_token_ptr5->static_check() == true);
    cout << arr_access_token_ptr5->run() << endl;
}

void test_data_type_token()
{
    print_red_str_to_term("test_data_type_token");
    shared_ptr<data_type_token> data_type_token_ptr(new data_type_token(UNSIGNED_LONG, true));

    cout << data_type_token_ptr->run() << endl;

    assert(data_type_token_ptr->static_check() == true);
}

void test_var_init_type_token()
{
    print_red_str_to_term("test_var_init_type_token");
    // 数据类型
    shared_ptr<data_type_token> data_type_token_ptr(new data_type_token(UNSIGNED_LONG, false));
    // 变量
    shared_ptr<var_name_token> init_var_name_ptr(new var_name_token("init_var_name", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> init_var_name_ptr2(new var_name_token("init_var_name", GLOBAL_MEM_VAR_TYPE));
    // 初始化表达式a
    shared_ptr<math_expr_token> init_math_express_ptr(new math_expr_token("init_math_express"));

    shared_ptr<var_init_token> var_init_type_token_ptr(new var_init_token(data_type_token_ptr, init_var_name_ptr, init_math_express_ptr));
    assert(var_init_type_token_ptr->static_check() == true);
    cout << var_init_type_token_ptr->run() << endl;

    shared_ptr<var_init_token> var_init_type_token_ptr2(new var_init_token(data_type_token_ptr, init_var_name_ptr2, init_math_express_ptr));
    assert(var_init_type_token_ptr2->static_check() == false);
}

void test_shared_mem_init_token()
{
    print_red_str_to_term("test_shared_mem_init_token");
    // 数据类型
    shared_ptr<data_type_token> data_type_declare(new data_type_token(UNSIGNED_LONG, false));
    // 共享内存类型的变量
    shared_ptr<var_name_token> init_shared_mem_var_name(new var_name_token("init_shared_mem_var_name", SHARED_MEM_VAR_TYPE));
    shared_ptr<var_name_token> init_shared_mem_var_name2(new var_name_token("init_shared_mem_var_name", GLOBAL_MEM_VAR_TYPE));

    // 共享内存大小的变量
    shared_ptr<var_name_token> shared_mem_size_var_name(new var_name_token("123", CONSTANT_VAR_TYPE));
    shared_ptr<var_name_token> shared_mem_size_var_name2(new var_name_token("123", GLOBAL_MEM_VAR_TYPE));

    shared_ptr<shared_mem_init_token> shared_mem_init_token_ptr(new shared_mem_init_token(data_type_declare, init_shared_mem_var_name, shared_mem_size_var_name));
    assert(shared_mem_init_token_ptr->static_check() == true);
    cout << shared_mem_init_token_ptr->run() << endl;

    shared_ptr<shared_mem_init_token> shared_mem_init_token_ptr2(new shared_mem_init_token(data_type_declare, init_shared_mem_var_name2, shared_mem_size_var_name));
    assert(shared_mem_init_token_ptr2->static_check() == false);

    shared_ptr<shared_mem_init_token> shared_mem_init_token_ptr3(new shared_mem_init_token(data_type_declare, init_shared_mem_var_name, shared_mem_size_var_name2));
    assert(shared_mem_init_token_ptr3->static_check() == false);
}

void test_shared_mem_write_token()
{
    print_red_str_to_term("test_shared_mem_write_token");
    // 创造共享内存变量
    shared_ptr<var_name_token> shared_mem_name(new var_name_token("shared_mem_name", SHARED_MEM_VAR_TYPE));
    shared_ptr<var_name_token> shared_mem_name2(new var_name_token("shared_mem_name", GLOBAL_MEM_VAR_TYPE));

    // 内存索引的变量
    shared_ptr<var_name_token> input_index(new var_name_token("123", CONSTANT_VAR_TYPE));
    shared_ptr<var_name_token> input_index2(new var_name_token("asd", GLOBAL_MEM_VAR_TYPE));

    // 写入数据的变量
    shared_ptr<var_name_token> written_value(new var_name_token("123", CONSTANT_VAR_TYPE));
    shared_ptr<var_name_token> written_value2(new var_name_token("asd", GLOBAL_MEM_VAR_TYPE));

    shared_ptr<shared_mem_write_token> shared_mem_write_token_ptr(new shared_mem_write_token(shared_mem_name, input_index, written_value));
    assert(shared_mem_write_token_ptr->static_check() == true);
    cout << shared_mem_write_token_ptr->run() << endl;

    shared_ptr<shared_mem_write_token> shared_mem_write_token_ptr2(new shared_mem_write_token(shared_mem_name2, input_index, written_value));
    assert(shared_mem_write_token_ptr2->static_check() == false);

    shared_ptr<shared_mem_write_token> shared_mem_write_token_ptr3(new shared_mem_write_token(shared_mem_name, input_index2, written_value));
    assert(shared_mem_write_token_ptr3->static_check() == false);

    shared_ptr<shared_mem_write_token> shared_mem_write_token_ptr4(new shared_mem_write_token(shared_mem_name, input_index, written_value2));
    assert(shared_mem_write_token_ptr4->static_check() == false);
}

// 测试for循环的抽象
void test_for_basic_token()
{
    print_red_str_to_term("test_for_basic_token");
    // 创造一个metadata get
    shared_ptr<metadata_get_basic_token> metadata_get_code(new metadata_get_basic_token(TBLOCK_META));
    shared_ptr<metadata_get_basic_token> metadata_get_code2(new metadata_get_basic_token(GLOBAL_META));

    // 创造inner loop的元数据
    shared_ptr<metadata_get_basic_token> inner_metadata_get_code(new metadata_get_basic_token(WARP_META));
    shared_ptr<metadata_get_basic_token> inner_metadata_get_code2(new metadata_get_basic_token(TBLOCK_META));

    // 创造一个inner loop
    shared_ptr<for_basic_token> inner_loop(new for_basic_token(WARP_META, inner_metadata_get_code, NULL, NULL, NULL));
    shared_ptr<for_basic_token> inner_loop2(new for_basic_token(TBLOCK_META, inner_metadata_get_code2, NULL, NULL, NULL));

    // 创造两个reduction
    shared_ptr<reduction_basic_token> reduction_code(new reduction_basic_token(TBLOCK_META));
    shared_ptr<reduction_basic_token> reduction_code2(new reduction_basic_token(WARP_META));

    shared_ptr<for_basic_token> loop(new for_basic_token(TBLOCK_META, metadata_get_code, inner_loop, reduction_code, NULL));
    cout << loop->run() << endl;
    assert(loop->static_check() == true);

    shared_ptr<for_basic_token> loop2(new for_basic_token(TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    cout << loop2->run() << endl;
    assert(loop2->static_check() == true);

    shared_ptr<for_basic_token> loop3(new for_basic_token(TBLOCK_META, metadata_get_code2, NULL, NULL, NULL));
    assert(loop3->static_check() == false);

    shared_ptr<for_basic_token> loop4(new for_basic_token(TBLOCK_META, metadata_get_code, inner_loop2, NULL, NULL));
    assert(loop4->static_check() == false);

    shared_ptr<for_basic_token> loop5(new for_basic_token(TBLOCK_META, metadata_get_code, inner_loop, reduction_code2, NULL));
    assert(loop5->static_check() == false);
}

// 正经for循环的测试
void test_for_token()
{
    print_red_str_to_term("test_for_token");
    // 创造一个metadata get
    shared_ptr<metadata_get_basic_token> metadata_get_code(new metadata_get_basic_token(TBLOCK_META));
    shared_ptr<metadata_get_basic_token> metadata_get_code2(new metadata_get_basic_token(WARP_META));

    // 创造迭代变量的类型
    shared_ptr<data_type_token> loop_var_name_type(new data_type_token(UNSIGNED_INT, false));
    shared_ptr<data_type_token> loop_var_name_type2(new data_type_token(UNSIGNED_INT, true));
    shared_ptr<data_type_token> loop_var_name_type3(new data_type_token(DOUBLE, false));

    // 创造迭代变量
    shared_ptr<var_name_token> loop_var_name(new var_name_token("id", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> loop_var_name2(new var_name_token("12", CONSTANT_VAR_TYPE));

    // 创造起始变量
    shared_ptr<var_name_token> begin_loop_var_name1(new var_name_token("12", CONSTANT_VAR_TYPE));
    shared_ptr<var_name_token> begin_loop_var_name2(new var_name_token("abc", GLOBAL_MEM_VAR_TYPE));

    // 创造结束变量
    shared_ptr<var_name_token> end_loop_var_name1(new var_name_token("12", CONSTANT_VAR_TYPE));
    shared_ptr<var_name_token> end_loop_var_name2(new var_name_token("abc", GLOBAL_MEM_VAR_TYPE));

    // 创造步长变量
    shared_ptr<var_name_token> step_loop_var_name1(new var_name_token("12", CONSTANT_VAR_TYPE));
    shared_ptr<var_name_token> step_loop_var_name2(new var_name_token("abc", GLOBAL_MEM_VAR_TYPE));

    // 创造真正的for循环
    shared_ptr<for_token> loop(new for_token(loop_var_name_type, loop_var_name, begin_loop_var_name1, end_loop_var_name1, step_loop_var_name1, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop->static_check() == true);
    cout << loop->run() << endl;

    shared_ptr<for_token> loop2(new for_token(loop_var_name_type2, loop_var_name, begin_loop_var_name1, end_loop_var_name1, step_loop_var_name1, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop2->static_check() == false);

    shared_ptr<for_token> loop3(new for_token(loop_var_name_type3, loop_var_name, begin_loop_var_name1, end_loop_var_name1, step_loop_var_name1, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop3->static_check() == false);

    shared_ptr<for_token> loop4(new for_token(loop_var_name_type, loop_var_name2, begin_loop_var_name1, end_loop_var_name1, step_loop_var_name1, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop4->static_check() == false);

    shared_ptr<for_token> loop5(new for_token(loop_var_name_type, loop_var_name, begin_loop_var_name2, end_loop_var_name1, step_loop_var_name1, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop5->static_check() == false);

    shared_ptr<for_token> loop6(new for_token(loop_var_name_type, loop_var_name, begin_loop_var_name1, end_loop_var_name2, step_loop_var_name1, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop6->static_check() == false);

    shared_ptr<for_token> loop7(new for_token(loop_var_name_type, loop_var_name, begin_loop_var_name1, end_loop_var_name1, step_loop_var_name2, TBLOCK_META, metadata_get_code, NULL, NULL, NULL));
    assert(loop7->static_check() == false);

    shared_ptr<for_token> loop8(new for_token(loop_var_name_type, loop_var_name, begin_loop_var_name1, end_loop_var_name1, step_loop_var_name1, TBLOCK_META, metadata_get_code2, NULL, NULL, NULL));
    assert(loop8->static_check() == false);
}

// 查看共享内存的广播
void test_shared_mem_broadcast_token()
{
    print_red_str_to_term("test_shared_mem_broadcast_token");
    // 创建一个数据类型
    shared_ptr<data_type_token> data_type_of_read_data(new data_type_token(UNSIGNED_LONG, false));
    shared_ptr<data_type_token> data_type_of_read_data2(new data_type_token(UNSIGNED_LONG, false));
    // 数组
    vector<shared_ptr<data_type_token>> data_type_of_read_data_vec;
    data_type_of_read_data_vec.push_back(data_type_of_read_data);
    data_type_of_read_data_vec.push_back(data_type_of_read_data2);

    // 创建共享内存数组的变量名称
    shared_ptr<var_name_token> global_mem_read_arr(new var_name_token("test1", GLOBAL_MEM_VAR_TYPE));
    shared_ptr<var_name_token> global_mem_read_arr2(new var_name_token("test2", GLOBAL_MEM_VAR_TYPE));
    // 数组
    vector<shared_ptr<var_name_token>> global_mem_read_arr_vec;
    global_mem_read_arr_vec.push_back(global_mem_read_arr);
    global_mem_read_arr_vec.push_back(global_mem_read_arr2);

    // 创建读取的索引
    shared_ptr<var_name_token> global_mem_read_index(new var_name_token("index1", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> global_mem_read_index2(new var_name_token("index2", REGISTER_VAR_TYPE));
    // 数组
    vector<shared_ptr<var_name_token>> global_mem_read_index_vec;
    global_mem_read_index_vec.push_back(global_mem_read_index);
    global_mem_read_index_vec.push_back(global_mem_read_index2);

    // 目标寄存器
    shared_ptr<var_name_token> dest_variable(new var_name_token("dest1", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> dest_variable2(new var_name_token("dest2", REGISTER_VAR_TYPE));
    // 数组
    vector<shared_ptr<var_name_token>> dest_variable_vec;
    dest_variable_vec.push_back(dest_variable);
    dest_variable_vec.push_back(dest_variable2);

    // 组装一个正确的共享内存广播
    shared_ptr<shared_mem_broadcast_token> shared_mem_broadcast_token_ptr(new shared_mem_broadcast_token(data_type_of_read_data_vec, global_mem_read_arr_vec, global_mem_read_index_vec, dest_variable_vec));
    assert(shared_mem_broadcast_token_ptr->static_check() == true);
    cout << shared_mem_broadcast_token_ptr->run() << endl;

    // 变量类型加入一个指针，肯定过不去
    shared_ptr<data_type_token> data_type_of_read_data3(new data_type_token(UNSIGNED_LONG, true));
    vector<shared_ptr<data_type_token>> data_type_of_read_data_vec2;
    data_type_of_read_data_vec2.push_back(data_type_of_read_data);
    data_type_of_read_data_vec2.push_back(data_type_of_read_data3);

    shared_ptr<shared_mem_broadcast_token> shared_mem_broadcast_token_ptr2(new shared_mem_broadcast_token(data_type_of_read_data_vec2, global_mem_read_arr_vec, global_mem_read_index_vec, dest_variable_vec));
    assert(shared_mem_broadcast_token_ptr2->static_check() == false);

    // 数组必须是GLOBAL数组
    shared_ptr<var_name_token> global_mem_read_arr3(new var_name_token("test3", SHARED_MEM_VAR_TYPE));
    vector<shared_ptr<var_name_token>> global_mem_read_arr_vec2;
    global_mem_read_arr_vec2.push_back(global_mem_read_arr);
    global_mem_read_arr_vec2.push_back(global_mem_read_arr3);

    shared_ptr<shared_mem_broadcast_token> shared_mem_broadcast_token_ptr3(new shared_mem_broadcast_token(data_type_of_read_data_vec, global_mem_read_arr_vec2, global_mem_read_index_vec, dest_variable_vec));
    assert(shared_mem_broadcast_token_ptr3->static_check() == false);

    // 索引必须是REGISTER类型
    shared_ptr<var_name_token> global_mem_read_index3(new var_name_token("index1", SHARED_MEM_VAR_TYPE));
    vector<shared_ptr<var_name_token>> global_mem_read_index_vec2;
    global_mem_read_index_vec2.push_back(global_mem_read_index);
    global_mem_read_index_vec2.push_back(global_mem_read_index3);

    shared_ptr<shared_mem_broadcast_token> shared_mem_broadcast_token_ptr4(new shared_mem_broadcast_token(data_type_of_read_data_vec, global_mem_read_arr_vec, global_mem_read_index_vec2, dest_variable_vec));
    assert(shared_mem_broadcast_token_ptr4->static_check() == false);

    // 目标寄存器必须是REGISTER类型
    shared_ptr<var_name_token> dest_variable3(new var_name_token("dest3", SHARED_MEM_VAR_TYPE));
    vector<shared_ptr<var_name_token>> dest_variable_vec2;
    dest_variable_vec2.push_back(dest_variable);
    dest_variable_vec2.push_back(dest_variable3);

    shared_ptr<shared_mem_broadcast_token> shared_mem_broadcast_token_ptr5(new shared_mem_broadcast_token(data_type_of_read_data_vec, global_mem_read_arr_vec, global_mem_read_index_vec, dest_variable_vec2));
    assert(shared_mem_broadcast_token_ptr5->static_check() == false);
}

void test_var_assign_token()
{
    print_red_str_to_term("test_var_assign_token");
    // 创建一个左操作数
    shared_ptr<var_name_token> left_operand1(new var_name_token("left_operand1", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> left_operand2(new var_name_token("left_operand2", CONSTANT_VAR_TYPE));

    // 创建变量类型的右操作数
    shared_ptr<var_name_token> right_operand1(new var_name_token("right_operand1", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> right_operand2(new var_name_token("right_operand2", GLOBAL_MEM_VAR_TYPE));

    // 创建一个算式类型的右操作数
    shared_ptr<math_expr_token> right_operand3(new math_expr_token("a+b"));

    // 创建合法的赋值语句
    shared_ptr<var_assign_token> var_assign1(new var_assign_token(left_operand1, right_operand1));
    assert(var_assign1->static_check() == true);

    // 创建合法的赋值语句
    shared_ptr<var_assign_token> var_assign2(new var_assign_token(left_operand1, right_operand3));
    assert(var_assign2->static_check() == true);

    // 创建非法的赋值语句
    shared_ptr<var_assign_token> var_assign3(new var_assign_token(left_operand2, right_operand1));
    assert(var_assign3->static_check() == false);

    shared_ptr<var_assign_token> var_assign4(new var_assign_token(left_operand1, right_operand2));
    assert(var_assign4->static_check() == false);
}

void test_metadata_set_get_token()
{
    print_red_str_to_term("test_metadata_set_get_token");
    // 一个初始化
    shared_ptr<data_type_token> data_type_token_ptr(new data_type_token(UNSIGNED_LONG, false));
    shared_ptr<var_name_token> init_var_name_ptr(new var_name_token("init_var_name", REGISTER_VAR_TYPE));
    shared_ptr<var_init_token> var_init_type_token_ptr(new var_init_token(data_type_token_ptr, init_var_name_ptr, NULL));

    // 一个共享内存广播
    // 创建一个数据类型
    shared_ptr<data_type_token> data_type_of_read_data(new data_type_token(UNSIGNED_LONG, false));
    // 数组
    vector<shared_ptr<data_type_token>> data_type_of_read_data_vec;
    data_type_of_read_data_vec.push_back(data_type_of_read_data);

    // 创建共享内存数组的变量名称
    shared_ptr<var_name_token> global_mem_read_arr(new var_name_token("test1", GLOBAL_MEM_VAR_TYPE));
    shared_ptr<var_name_token> global_mem_read_arr2(new var_name_token("test2", GLOBAL_MEM_VAR_TYPE));
    // 数组
    vector<shared_ptr<var_name_token>> global_mem_read_arr_vec;
    global_mem_read_arr_vec.push_back(global_mem_read_arr);

    // 创建读取的索引
    shared_ptr<var_name_token> global_mem_read_index(new var_name_token("index1", REGISTER_VAR_TYPE));
    // 数组
    vector<shared_ptr<var_name_token>> global_mem_read_index_vec;
    global_mem_read_index_vec.push_back(global_mem_read_index);

    // 目标寄存器
    shared_ptr<var_name_token> dest_variable(new var_name_token("dest1", REGISTER_VAR_TYPE));
    // 数组
    vector<shared_ptr<var_name_token>> dest_variable_vec;
    dest_variable_vec.push_back(dest_variable);

    // 组装一个正确的共享内存广播
    shared_ptr<shared_mem_broadcast_token> shared_mem_broadcast_token_ptr(new shared_mem_broadcast_token(data_type_of_read_data_vec, global_mem_read_arr_vec, global_mem_read_index_vec, dest_variable_vec));

    // 一个计算
    // 创建一个左操作数
    shared_ptr<var_name_token> left_operand1(new var_name_token("left_operand1", REGISTER_VAR_TYPE));

    // 创建一个算式类型的右操作数
    shared_ptr<math_expr_token> right_operand3(new math_expr_token("a+b"));

    // 创建合法的赋值语句
    shared_ptr<var_assign_token> var_assign1(new var_assign_token(left_operand1, right_operand3));

    // 一个全局内存访问
    shared_ptr<var_name_token> dest_var_token_ptr(new var_name_token("dest_var", REGISTER_VAR_TYPE));

    shared_ptr<var_name_token> mem_ptr_name_token_ptr2(new var_name_token("mem_ptr_name", GLOBAL_MEM_VAR_TYPE));

    shared_ptr<var_name_token> mem_index_token_ptr(new var_name_token("mem_index", REGISTER_VAR_TYPE));

    // 创造一个新的global mem赋值的语句
    shared_ptr<arr_access_token> arr_access_token_ptr(new arr_access_token(dest_var_token_ptr, mem_ptr_name_token_ptr2, mem_index_token_ptr));

    // 合法的TBLOCK级别的元数据获取
    shared_ptr<metadata_set_get_token> metadata_set_get_token_ptr(new metadata_set_get_token(TBLOCK_META));

    metadata_set_get_token_ptr->add_metadata_get_expr(var_init_type_token_ptr);
    metadata_set_get_token_ptr->add_metadata_get_expr(shared_mem_broadcast_token_ptr);
    metadata_set_get_token_ptr->add_metadata_get_expr(var_assign1);
    metadata_set_get_token_ptr->add_metadata_get_expr(arr_access_token_ptr);

    assert(metadata_set_get_token_ptr->static_check() == true);
    cout << metadata_set_get_token_ptr->run() << endl;

    // 不合法的WARP级别的元数据获取
    shared_ptr<metadata_set_get_token> metadata_set_get_token_ptr2(new metadata_set_get_token(WARP_META));

    metadata_set_get_token_ptr2->add_metadata_get_expr(var_init_type_token_ptr);
    metadata_set_get_token_ptr2->add_metadata_get_expr(shared_mem_broadcast_token_ptr);
    metadata_set_get_token_ptr2->add_metadata_get_expr(var_assign1);
    metadata_set_get_token_ptr2->add_metadata_get_expr(arr_access_token_ptr);

    assert(metadata_set_get_token_ptr2->static_check() == false);
}

void test_var_of_metadata_from_spec_paral()
{
    print_red_str_to_term("test_var_of_metadata_from_spec_paral");
    // 创造一个表达式
    shared_ptr<math_expr_token> idx_expr(new math_expr_token("a+b"));

    cout << var_of_metadata_from_spec_paral(TBLOCK_META, "metadata", 1, idx_expr) << endl;
}

// 测试胶水代码
void test_basic_glue_code()
{
    print_red_str_to_term("test_basic_glue_code");
    // 首先初始化两个IO的结构
    shared_ptr<basic_IO_of_reduction> input_IO(new basic_IO_of_reduction("total_thread_result", THREAD_META));
    shared_ptr<basic_IO_of_reduction> output_IO(new basic_IO_of_reduction("shared_mem_result", TBLOCK_META));

    // 输入变量
    shared_ptr<var_name_token> input_var(new var_name_token("input_var", REGISTER_VAR_TYPE));
    shared_ptr<var_name_token> output_var(new var_name_token("output_var", SHARED_MEM_VAR_TYPE));

    input_IO->add_var_name(input_var);
    output_IO->add_var_name(output_var);

    // 用输入和输出来初始化一个胶水
    shared_ptr<basic_glue_code> basic_glue_code_ptr(new basic_glue_code(input_IO, output_IO));
    // 打印对应的内筒
    cout << basic_glue_code_ptr->run() << endl;
}

// 检查格式的输出
void test_format_output_of_code_generator()
{
    print_red_str_to_term("test_format_output_of_code_generator");
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 执行一次BMTB的行切分
    shared_ptr<basic_operator> fixed_interval_row_direction_tblock_blocking_operator_ptr(new fixed_interval_row_direction_tblock_blocking_operator(meta_dataset_ptr, 2, 512, true));

    // 执行
    fixed_interval_row_direction_tblock_blocking_operator_ptr->run();

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    // 代码生成器
    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 添加三个数据，BMTB行偏移量、类索引、值索引
    code_generator_ptr->add_new_metadata_dependency(GLOBAL_META, "nz_col_indices", 2);
    code_generator_ptr->add_new_metadata_dependency(GLOBAL_META, "nz_vals", 2);
    code_generator_ptr->add_new_metadata_dependency(TBLOCK_META, "first_row_indices", 2);

    // 将数据写到内存中
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 检查文件是否存在
    assert(file_is_exist(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)));
    assert(file_is_exist(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id) + "/" + get_metadata_item_name(GLOBAL_META, "nz_col_indices", 2)));
    assert(file_is_exist(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id) + "/" + get_metadata_item_name(GLOBAL_META, "nz_vals", 2)));
    assert(file_is_exist(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id) + "/" + get_metadata_item_name(TBLOCK_META, "first_row_indices", 2)));

    // 移除文件
    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试对于文件的输出
void test_code_generator_code_file_output_test()
{
    print_red_str_to_term("test_code_generator_code_file_output_test");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 执行一次BMTB的行切分
    shared_ptr<basic_operator> fixed_interval_row_direction_tblock_blocking_operator_ptr(new fixed_interval_row_direction_tblock_blocking_operator(meta_dataset_ptr, 2, 512, true));

    // 执行
    fixed_interval_row_direction_tblock_blocking_operator_ptr->run();

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    // 代码生成器
    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 添加三个数据，BMTB行偏移量、类索引、值索引
    code_generator_ptr->add_new_metadata_dependency(GLOBAL_META, "nz_col_indices", 2);
    code_generator_ptr->add_new_metadata_dependency(GLOBAL_META, "nz_vals", 2);
    code_generator_ptr->add_new_metadata_dependency(TBLOCK_META, "first_row_indices", 2);

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建一个文件
    code_generator_ptr->generate_kernel_file(10);

    // 移除文件
    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试内核代码头部对于线程网格信息的计算是不是正确
void test_code_generator_thread_grid_info_get_in_the_beginning_of_kernel()
{
    print_red_str_to_term("test_code_generator_thread_grid_info_get_in_the_beginning_of_kernel");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    // 代码生成器
    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 添加值和线程
    code_generator_ptr->add_new_metadata_dependency(GLOBAL_META, "nz_col_indices", 2);
    code_generator_ptr->add_new_metadata_dependency(GLOBAL_META, "nz_vals", 2);

    // 打印几个grid信息
    cout << code_generator_ptr->global_warp_id_code() << endl;
    cout << code_generator_ptr->global_thread_id_code() << endl;
    cout << code_generator_ptr->global_thread_block_id_code() << endl;
    cout << code_generator_ptr->total_thread_block_num_code() << endl;
    cout << code_generator_ptr->total_warp_num_code() << endl;
    cout << code_generator_ptr->total_thread_num_code() << endl;
    cout << code_generator_ptr->thread_id_in_thread_block_code() << endl;
    cout << code_generator_ptr->warp_id_in_thread_block_code() << endl;
    cout << code_generator_ptr->thread_id_in_warp_code() << endl;
    cout << code_generator_ptr->warp_num_in_thread_block_code() << endl;
    cout << code_generator_ptr->thread_num_in_thread_block_code() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建一个文件
    code_generator_ptr->generate_kernel_file(10);

    // 移除文件
    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试不合并的元数据访存代码
void test_unfused_memory_access()
{
    // 首先切一下矩阵
    print_red_str_to_term("test_unfused_memory_access");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    shared_ptr<math_expr_token> mem_access_index1(new math_expr_token("0"));
    shared_ptr<math_expr_token> mem_access_index2(new math_expr_token("1"));

    // 增加不能合并的访问
    vector<shared_ptr<basic_token>> mem_access_code_1 = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_col_indices", mem_access_index1, false);
    vector<shared_ptr<basic_token>> mem_access_code_2 = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_vals", mem_access_index2, false);

    cout << mem_access_code_1[0]->run() << endl;
    cout << mem_access_code_1[1]->run() << endl;
    cout << mem_access_code_2[0]->run() << endl;
    cout << mem_access_code_2[1]->run() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试合并的访存的代码
void test_fused_metadata_get_token()
{
    // 首先切一下矩阵
    print_red_str_to_term("test_fused_metadata_get_token");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    // 创建一个代码生成器
    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    shared_ptr<math_expr_token> mem_access_index1(new math_expr_token("0"));
    shared_ptr<math_expr_token> mem_access_index2(new math_expr_token("1"));

    // 创建两个合并的访存
    shared_ptr<var_name_token> read_data_var_name1 = code_generator_ptr->generate_fused_memory_access(GLOBAL_META, "nz_col_indices", mem_access_index1);
    shared_ptr<var_name_token> read_data_var_name2 = code_generator_ptr->generate_fused_memory_access(GLOBAL_META, "nz_vals", mem_access_index2);

    cout << read_data_var_name1->run() << endl;
    cout << read_data_var_name2->run() << endl;

    // 输出GLOBAL的所有合并访存
    shared_ptr<metadata_set_get_token> metadata_set_get_token_ptr = code_generator_ptr->generate_token_of_fused_metadata_get_in_spec_paral_level(GLOBAL_META);

    cout << metadata_set_get_token_ptr->run() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试for循环的构造token，以BMBT为最外层
void test_outest_for_token_of_BMBT()
{
    print_red_str_to_term("test_outest_for_token_of_BMBT");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 创建一个线程块粒度的切分
    shared_ptr<basic_operator> fixed_interval_row_direction_tblock_blocking_operator_ptr(new fixed_interval_row_direction_tblock_blocking_operator(meta_dataset_ptr, 2, 512, true));

    // 执行
    fixed_interval_row_direction_tblock_blocking_operator_ptr->run();

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 加入tblock级别的并行
    code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);

    // 生成对应级别for循环
    shared_ptr<for_token> for_token_ptr = code_generator_ptr->generate_for_token_of_spec_paral_level(TBLOCK_META, NULL, true);

    cout << for_token_ptr->run() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试for循环的构造token，以BMW为最外层
void test_outest_for_token_of_BMW()
{
    print_red_str_to_term("test_outest_for_token_of_BMW");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 创建一个WARP粒度的切分
    shared_ptr<basic_operator> fixed_interval_row_direction_warp_blocking_operator_ptr(new fixed_interval_row_direction_warp_blocking_operator(meta_dataset_ptr, 2, 512, false, false, true));
    fixed_interval_row_direction_warp_blocking_operator_ptr->run();

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 加入tblock级别的并行
    code_generator_ptr->open_spec_level_of_paral(WARP_META);

    // 生成对应级别for循环
    shared_ptr<for_token> for_token_ptr = code_generator_ptr->generate_for_token_of_spec_paral_level(WARP_META, NULL, true);

    cout << for_token_ptr->run() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试for循环的构造token，以BMT为最外层
void test_outest_for_token_of_BMT()
{
    print_red_str_to_term("test_outest_for_token_of_BMT");

    // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/IG5-18.mtx.coo", "IG5-18");
    // shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file("/home/guoxiao/spmv_builder/data_source/IG5-18.mtx.coo", "IG5-18");
    shared_ptr<operator_executer> executer(new operator_executer());

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 10000, executer->get_operator_context()));
    executer->add_and_run(test_fixed_interval_row_matrix_div_operator_ptr);

    shared_ptr<basic_operator> fixed_interval_nnz_direction_warp_blocking_operator_ptr(new fixed_interval_nnz_direction_warp_blocking_operator(meta_dataset_ptr, 1, 160, false, false, true, executer->get_operator_context()));
    // 执行
    executer->add_and_run(fixed_interval_nnz_direction_warp_blocking_operator_ptr);

    shared_ptr<basic_operator> fixed_interval_nnz_direction_thread_blocking_operator_ptr(new fixed_interval_nnz_direction_thread_blocking_operator(meta_dataset_ptr, 1, 5, true, true, false, executer->get_operator_context()));
    // 执行
    executer->add_and_run(fixed_interval_nnz_direction_thread_blocking_operator_ptr);

    vector<shared_ptr<transform_step_record_item>> record_vec = executer->get_transform_history();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 1));

    // 加入thread级别的并行
    code_generator_ptr->open_spec_level_of_paral(THREAD_META);

    // 生成对应级别for循环
    shared_ptr<for_token> for_token_ptr = code_generator_ptr->generate_for_token_of_spec_paral_level(THREAD_META, NULL, true);

    cout << for_token_ptr->run() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 处理内层循环，先TBLOCK级别切分，然后THREAD级别切分
void test_inner_loop_for_token()
{
    print_red_str_to_term("test_inner_loop_for_token");
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "IG5-15");
    shared_ptr<operator_executer> executer(new operator_executer());

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000, executer->get_operator_context()));

    // 执行
    executer->add_and_run(test_fixed_interval_row_matrix_div_operator_ptr);

    // 然后执行一个BMTB分块
    shared_ptr<basic_operator> test_fix_interval_row_direction_tblock_blocking_operator_ptr(new fixed_interval_row_direction_tblock_blocking_operator(meta_dataset_ptr, 2, 128, false, executer->get_operator_context()));

    // 执行
    executer->add_and_run(test_fix_interval_row_direction_tblock_blocking_operator_ptr);

    shared_ptr<basic_operator> test_fixed_interval_row_direction_warp_blocking_operator_ptr(new fixed_interval_row_direction_warp_blocking_operator(meta_dataset_ptr, 2, 64, false, false, false, executer->get_operator_context()));

    executer->add_and_run(test_fixed_interval_row_direction_warp_blocking_operator_ptr);

    shared_ptr<basic_operator> test_balanced_interval_row_direction_thread_blocking_operator_ptr(new balanced_interval_row_direction_thread_blocking_operator(meta_dataset_ptr, 2, 32, true, true, executer->get_operator_context()));
    executer->add_and_run(test_balanced_interval_row_direction_thread_blocking_operator_ptr);

    vector<shared_ptr<transform_step_record_item>> record_vec = executer->get_transform_history();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }

    assert(logical_check(meta_dataset_ptr) == true);

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 加入thread级别的并行
    code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);
    code_generator_ptr->open_spec_level_of_paral(WARP_META);

    // 生成对应级别for循环
    shared_ptr<for_token> for_token_ptr = code_generator_ptr->generate_for_token_of_spec_paral_level(WARP_META, NULL, false);

    cout << for_token_ptr->run() << endl;

    // 获取TBLOCK级别元数据获取的代码
    shared_ptr<metadata_set_get_token> metadata_token_ptr = code_generator_ptr->generate_token_of_fused_metadata_get_in_spec_paral_level(TBLOCK_META);

    cout << metadata_token_ptr->run() << endl;

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}

// 测试共享内存的声明支持
void test_shared_memory_array_declaration()
{
    print_red_str_to_term("test_shared_memory_array_declaration");
     // 首先创建一个Operator graph
    shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["ROOT_PATH_STR"].as_string() + "/data_source/in-2004.mtx.coo", "in-2004");

    shared_ptr<basic_operator> test_fixed_interval_row_matrix_div_operator_ptr(new fixed_interval_row_matrix_div_operator(meta_dataset_ptr, 0, 500000));

    cout << test_fixed_interval_row_matrix_div_operator_ptr->is_valid_according_to_metadata() << endl;

    // 执行
    test_fixed_interval_row_matrix_div_operator_ptr->run();

    cout << test_fixed_interval_row_matrix_div_operator_ptr->convert_to_string() << endl;

    for (int i = 0; i < test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence().size(); i++)
    {
        cout << test_fixed_interval_row_matrix_div_operator_ptr->get_data_transform_sequence()[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 2));

    // 注册多个共享内存
    code_generator_ptr->add_new_use_of_shared_mem(UNSIGNED_LONG, "test", 1);
    assert(code_generator_ptr->shared_mem_is_exist("test"));
    code_generator_ptr->add_new_use_of_shared_mem(UNSIGNED_LONG, "test2", 1);

    // 创建format文件
    unsigned long output_id = code_generator_ptr->write_matrix_format_to_disk();

    // 创建代码文件
    code_generator_ptr->generate_kernel_file(10);

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
}



float test_spmm_thread_total(shared_ptr<meta_data_set>  meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_thread_total");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());


    shared_ptr<basic_operator> sort_operator_ptr(new sort_operator(code_generator_ptr, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(sort_operator_ptr);

    cout << sort_operator_ptr->convert_to_string() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec_s = sort_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec_s.size(); i++)
    {
        cout << record_vec_s[i]->convert_to_string() << endl;
    }


    // 执行一次BMT的行切分
    shared_ptr<basic_operator> fixed_interval_row_direction_thread_blocking_operator_ptr(new fixed_interval_row_direction_thread_blocking_operator(code_generator_ptr, 1, false, false, false, false, true, sparse_coarsen_factor, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_row_direction_thread_blocking_operator_ptr);

    cout << fixed_interval_row_direction_thread_blocking_operator_ptr->convert_to_string() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_row_direction_thread_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }



    int x_size;
    if ((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor < 32)
    {
        x_size = (int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor;
        set_config("VECTOR_WIDTH", x_size);

    }
    else
    {
        x_size = 32;
        set_config("VECTOR_WIDTH", 32);

    }

    int y_size = 128 / x_size;


    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    // 创建一个reduction token
    shared_ptr<thread_total_reduce_operator> thread_reduce_operator_ptr(new thread_total_reduce_operator(code_generator_ptr, false, sparse_coarsen_factor, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(thread_reduce_operator_ptr);


    vector<unsigned int> block;


    block.push_back(x_size);
    block.push_back(y_size);

    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)->get_metadata_arr()->read_integer_from_arr(0) / y_size + 1;

    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);
    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_thread_total_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;
}




// float test_spmm_thread_total_with_col_div(shared_ptr<meta_data_set>  meta_dataset_ptr,string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
// {
//     print_red_str_to_term("test_spmm_thread_total");

//     shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

//     shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());


//     shared_ptr<basic_operator> sort_operator_ptr(new sort_operator(code_generator_ptr, operator_executer_ptr->get_operator_context()));

//     operator_executer_ptr->add_and_run(sort_operator_ptr);

//     cout << sort_operator_ptr->convert_to_string() << endl;

//     vector<shared_ptr<transform_step_record_item>> record_vec_s = sort_operator_ptr->get_data_transform_sequence();

//     for (int i = 0; i < record_vec_s.size(); i++)
//     {
//         cout << record_vec_s[i]->convert_to_string() << endl;
//     }


//     // 执行一次BMT的行切分
//     shared_ptr<basic_operator> fixed_interval_col_direction_thread_blocking_operator_ptr(new fixed_interval_col_direction_thread_blocking_operator(code_generator_ptr, 128, false, false, true, false, operator_executer_ptr->get_operator_context()));

//     operator_executer_ptr->add_and_run(fixed_interval_col_direction_thread_blocking_operator_ptr);

//     cout << fixed_interval_col_direction_thread_blocking_operator_ptr->convert_to_string() << endl;

//     vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_col_direction_thread_blocking_operator_ptr->get_data_transform_sequence();

//     for (int i = 0; i < record_vec.size(); i++)
//     {
//         cout << record_vec[i]->convert_to_string() << endl;
//     }

//     // 执行完之后Metadata set中的内容是合法的
//     assert(meta_dataset_ptr->check());
//     assert(logical_check(meta_dataset_ptr) == true);

//     // 创建一个reduction token
//     shared_ptr<thread_total_reduce_operator> thread_reduce_operator_ptr(new thread_total_reduce_operator(code_generator_ptr, false, 1 , 1, operator_executer_ptr->get_operator_context()));
//     operator_executer_ptr->add_and_run(thread_reduce_operator_ptr);


//     vector<unsigned int> block;


//     int x_size;
//     if ((int)get_config()["DENSE_MATRIX_SIZE"].as_float() < 32)
//     {
//         x_size = (int)get_config()["DENSE_MATRIX_SIZE"].as_float();
//         set_config("VECTOR_WIDTH", x_size);

//     }
//     else
//     {
//         x_size = 32;
//     }

//     int y_size = 512 / x_size;

//     block.push_back(x_size);
//     block.push_back(y_size);
//     unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)->get_metadata_arr()->read_integer_from_arr(0) / y_size + 1;

//     shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, 1, operator_executer_ptr->get_operator_context()));
//     operator_executer_ptr->add_and_run(grid_block_operator_ptr);

//     code_generator_ptr->compile();

//     unsigned long output_id = code_generator_ptr->generate_final_program(1000);
//     string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
//     float time = 0;
//     float gflops = 0;
//     execute_binary(execute_path, matrix_name, time, gflops);
//     cout << matrix_name << " " << gflops << endl;
//     ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_thread_total_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
//     out << matrix_name<< " " << gflops << endl;

//     // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
//     return gflops;

// }






float test_spmm_warp_total(shared_ptr<meta_data_set> meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_warp_total");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    // 执行一次BMT的行切分
    shared_ptr<basic_operator> fixed_interval_row_direction_warp_blocking_operator_ptr(
        new fixed_interval_row_direction_warp_blocking_operator(code_generator_ptr, 1, false, false, false, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_row_direction_warp_blocking_operator_ptr);

    cout << fixed_interval_row_direction_warp_blocking_operator_ptr->convert_to_string() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_row_direction_warp_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)->get_metadata_arr()->read_integer_from_arr(0);
    vector<unsigned int> block;

    int y_size = min((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor, 32);
    int x_size = max(256 / y_size, 32);
    block.push_back(x_size);
    block.push_back(y_size);

    set_config("VECTOR_WIDTH", x_size);
    // 创建一个reduction token
    shared_ptr<warp_total_reduce_operator> warp_reduce_operator_ptr(
        new warp_total_reduce_operator(code_generator_ptr, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(warp_reduce_operator_ptr);



    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);

    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_warp_total_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;

}



float test_spmm_warp_bitmap(shared_ptr<meta_data_set> meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_warp_bitmap");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    // 执行一次BMT的行切分
    shared_ptr<basic_operator> fixed_interval_col_direction_thread_blocking_operator_ptr(
        new fixed_interval_col_direction_thread_blocking_operator(code_generator_ptr, 64, false, false, true, false, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_col_direction_thread_blocking_operator_ptr);

    cout << operator_executer_ptr->get_operator_context()->read_operator_context_arr(DISTRIBUTING_OP, 0)[0]->get_name() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_col_direction_thread_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)->get_metadata_arr()->read_integer_from_arr(0);
    int y_size = min((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor, 32);
    int x_size = max(128 / y_size, 32);
    set_config("VECTOR_WIDTH", x_size);

    shared_ptr<thread_total_reduce_operator> thread_total_reduce_operator_ptr(new thread_total_reduce_operator(code_generator_ptr, true, sparse_coarsen_factor, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(thread_total_reduce_operator_ptr);

    // 创建一个reduction token
    shared_ptr<warp_bit_map_operator> warp_reduce_operator_ptr(new warp_bit_map_operator(code_generator_ptr, coarsen_factor, true, true, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(warp_reduce_operator_ptr);


    vector<unsigned int> block;

    block.push_back(x_size);
    block.push_back(y_size);

    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);
    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_warp_bitmap_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;

}



float test_spmm_thread_bit_map(shared_ptr<meta_data_set> meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_thread_bit_map");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    // 执行一次BMT的行切分
    shared_ptr<basic_operator> fixed_interval_nnz_direction_thread_blocking_operator_ptr(new fixed_interval_nnz_direction_thread_blocking_operator(code_generator_ptr, 32, false, false, true, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_nnz_direction_thread_blocking_operator_ptr);

    cout << fixed_interval_nnz_direction_thread_blocking_operator_ptr->convert_to_string() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_nnz_direction_thread_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }
    vector<unsigned int> block;

    int x_size;
    if ((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor < 32)
    {
        x_size = (int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor;
        set_config("VECTOR_WIDTH", x_size);

    }
    else
    {
        x_size = 32;
        set_config("VECTOR_WIDTH", 32);

    }
    int y_size = 128 / x_size;

    block.push_back(x_size);
    block.push_back(y_size);
    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_nnz_num", -1)->get_metadata_arr()->read_integer_from_arr(0) / y_size + 1;

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);
    // 存在两个层次的并行
    // code_generator_ptr->open_spec_level_of_paral(TBLOCK_META);
    code_generator_ptr->open_spec_level_of_paral(THREAD_META);

    // 创建一个reduction token
    shared_ptr<thread_bit_map_operator> thread_reduce_operator_ptr(new thread_bit_map_operator(code_generator_ptr, THREAD_META, x_size, sparse_coarsen_factor, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(thread_reduce_operator_ptr);



    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);
    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_thread_bit_map_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;

}

float test_spmm_warp_segment(shared_ptr<meta_data_set> meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_warp_segment");


    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    // 执行一次BMT的行切分
    shared_ptr<basic_operator> fixed_interval_nnz_direction_thread_blocking_operator_ptr(new fixed_interval_nnz_direction_thread_blocking_operator(code_generator_ptr, 32, false, false, true, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_nnz_direction_thread_blocking_operator_ptr);

    cout << operator_executer_ptr->get_operator_context()->read_operator_context_arr(DISTRIBUTING_OP, 0)[0]->get_name() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_nnz_direction_thread_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);


    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_nnz_num", -1)->get_metadata_arr()->read_integer_from_arr(0) / 128 + 1;
    vector<unsigned int> block;

    int x_size = min((int)get_config()["DENSE_MATRIX_SIZE"].as_float(), 32);
    int y_size = 256 / x_size;
    block.push_back(x_size);
    block.push_back(y_size);
    set_config("VECTOR_WIDTH", x_size);
    shared_ptr<thread_bit_map_operator> thread_total_reduce_operator_ptr(new thread_bit_map_operator(code_generator_ptr, WARP_META, get_config()["VECTOR_WIDTH"].as_float(), sparse_coarsen_factor, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(thread_total_reduce_operator_ptr);

    // 创建一个reduction token
    shared_ptr<warp_segment_reduce_operator> warp_reduce_operator_ptr(new warp_segment_reduce_operator(code_generator_ptr, coarsen_factor, false, false, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(warp_reduce_operator_ptr);



    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);
    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_warp_segment_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;

}


float test_spmm_block_total(shared_ptr<meta_data_set> meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_block_total");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    shared_ptr<basic_operator> fixed_interval_row_direction_tblock_blocking_operator_ptr(new fixed_interval_row_direction_tblock_blocking_operator(code_generator_ptr, 1, false, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_row_direction_tblock_blocking_operator_ptr);

    cout << operator_executer_ptr->get_operator_context()->read_operator_context_arr(DISTRIBUTING_OP, 0)[0]->get_name() << endl;

    vector<shared_ptr<transform_step_record_item>> record_vec = fixed_interval_row_direction_tblock_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec.size(); i++)
    {
        cout << record_vec[i]->convert_to_string() << endl;
    }

    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);


    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)->get_metadata_arr()->read_integer_from_arr(0);
    vector<unsigned int> block;

    int x_size = min((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / 1, 32);
    int y_size =  256 / x_size;

    block.push_back(x_size);
    block.push_back(y_size);
    set_config("VECTOR_WIDTH", x_size);

    shared_ptr<tblock_total_reduce_operator> block_reduce_operator_ptr(new tblock_total_reduce_operator(code_generator_ptr, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(block_reduce_operator_ptr);

    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);
    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_block_total_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;

}



float test_spmm_block_bitmap(shared_ptr<meta_data_set> meta_dataset_ptr, string matrix_name, int sparse_coarsen_factor, int coarsen_factor)
{
    print_red_str_to_term("test_spmm_block_bitmap");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));

    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    shared_ptr<basic_operator> fixed_interval_col_direction_thread_blocking_operator_ptr(new fixed_interval_col_direction_thread_blocking_operator(code_generator_ptr, 64, false, false, true, false, operator_executer_ptr->get_operator_context()));

    operator_executer_ptr->add_and_run(fixed_interval_col_direction_thread_blocking_operator_ptr);

    vector<shared_ptr<transform_step_record_item>> record_vec_thread = fixed_interval_col_direction_thread_blocking_operator_ptr->get_data_transform_sequence();

    for (int i = 0; i < record_vec_thread.size(); i++)
    {
        cout << record_vec_thread[i]->convert_to_string() << endl;
    }



    // 执行完之后Metadata set中的内容是合法的
    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)->get_metadata_arr()->read_integer_from_arr(0);
    vector<unsigned int> block;


    int x_size = min((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor, 32);
    int y_size = 256 / x_size;

    block.push_back(x_size);
    block.push_back(y_size);

    set_config("VECTOR_WIDTH", x_size);

    shared_ptr<thread_total_reduce_operator> thread_total_reduce_operator_ptr(new thread_total_reduce_operator(code_generator_ptr, false, sparse_coarsen_factor, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(thread_total_reduce_operator_ptr);


    shared_ptr<grid_block_operator> grid_block_operator_ptr(new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    shared_ptr<tblock_thread_bit_map_operator> block_reduce_operator_ptr(new tblock_thread_bit_map_operator(code_generator_ptr, coarsen_factor, block[1], false, false, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(block_reduce_operator_ptr);




    code_generator_ptr->compile();

    unsigned long output_id = code_generator_ptr->generate_final_program(10000);
    string execute_path = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    float time = 0;
    float gflops = 0;
    execute_binary(execute_path, matrix_name, time, gflops);
    cout << matrix_name << " " << gflops << endl;
    ofstream out("/home/wangyaoyu/GeneralSparse/test_spmm_block_bitmap_" + get_config()["DENSE_MATRIX_SIZE"].as_string() + ".log", ios::app);
    out << matrix_name<< " " << gflops << endl;

    // system(("rm -rf " + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id)).c_str());
    return time / 10000;

}







void test_all()
{
    test_math_expr_token();
    test_var_name_token();
    test_arr_access_token();
    test_data_type_token();
    test_var_init_type_token();
    test_shared_mem_init_token();
    test_shared_mem_write_token();
    test_for_basic_token();
    test_for_token();
    test_shared_mem_broadcast_token();
    test_var_assign_token();
    test_metadata_set_get_token();
    test_var_of_metadata_from_spec_paral();
    test_basic_glue_code();
    test_format_output_of_code_generator();
    test_code_generator_code_file_output_test();
    test_code_generator_thread_grid_info_get_in_the_beginning_of_kernel();
    test_unfused_memory_access();
    test_fused_metadata_get_token();
    test_outest_for_token_of_BMBT();
    test_outest_for_token_of_BMW();
    test_outest_for_token_of_BMT();
    test_inner_loop_for_token();
    test_shared_memory_array_declaration();
//     test_global_var_declaration();
//     test_total_BMT_result_reduce_to_one_register_token();
//     test_for_structure_build();
//     test_insert_reduction_token_to_for_token();
//     test_insert_metadata_get_token();
//     test_complete_kernel_output();
// }
}

int main(int argc, char ** argv)
{
    string matrix_name = argv[1];
    string matrix_size = argv[2];

    set_config("DENSE_MATRIX_SIZE", stoi(matrix_size));
    float time = 0;
    float min_time = 999999;

    shared_ptr<meta_data_set> meta_dataset_ptr;
    // shared_ptr<meta_data_set> meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_warp_bitmap(meta_dataset_ptr, matrix_name, 4, 1);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_block_total(meta_dataset_ptr, matrix_name, 1, 1);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_block_bitmap(meta_dataset_ptr, matrix_name, 4, 1);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    time = test_spmm_thread_total(meta_dataset_ptr, matrix_name, 4, 1);
    if (time < min_time)
    {
        min_time = time;
    }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_thread_bit_map(meta_dataset_ptr, matrix_name, 4, 1);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_warp_bitmap(meta_dataset_ptr, matrix_name, 4, 2);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_block_bitmap(meta_dataset_ptr, matrix_name, 4, 2);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_thread_total(meta_dataset_ptr, matrix_name, 4, 2);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_thread_bit_map(meta_dataset_ptr, matrix_name, 4, 2);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_block_total(meta_dataset_ptr, matrix_name, 1, 4);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_block_total(meta_dataset_ptr, matrix_name, 1, 2);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }

    // meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    // time = test_spmm_warp_total(meta_dataset_ptr, matrix_name, 1, 1);
    // if (time < min_time)
    // {
    //     min_time = time;
    // }


    // // 将global_config 文件中HALF置为false, 保持PRECISE_OF_FLOAT为float即可测试float
    // if (get_config()["HALF"].as_bool() == true)
    // {

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_bit_map(meta_dataset_ptr, matrix_name, 8, 2);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_total(meta_dataset_ptr, matrix_name, 8, 2);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_bitmap(meta_dataset_ptr, matrix_name, 8, 2);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_block_bitmap(meta_dataset_ptr, matrix_name, 8, 2);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_segment(meta_dataset_ptr, matrix_name, 4, 2);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_total(meta_dataset_ptr, matrix_name, 8, 4);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_bit_map(meta_dataset_ptr, matrix_name, 8, 4);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_bitmap(meta_dataset_ptr, matrix_name, 8, 4);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_segment(meta_dataset_ptr, matrix_name, 8, 4);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_total(meta_dataset_ptr, matrix_name, 4, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_bit_map(meta_dataset_ptr, matrix_name, 4, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_bitmap(meta_dataset_ptr, matrix_name, 4, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_segment(meta_dataset_ptr, matrix_name, 4, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_total(meta_dataset_ptr, matrix_name, 8, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_thread_bit_map(meta_dataset_ptr, matrix_name, 8, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_bitmap(meta_dataset_ptr, matrix_name, 8, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }

    //     meta_dataset_ptr = create_init_metadata_set_from_file(get_config()["DATA_SET"].as_string() + matrix_name + "", "");
    //     time = test_spmm_warp_segment(meta_dataset_ptr, matrix_name, 8, 8);
    //     if (time < min_time)
    //     {
    //         min_time = time;
    //     }
    // }

    cout << matrix_name << " : " << "min time: " << min_time << endl;

    return 0;
}