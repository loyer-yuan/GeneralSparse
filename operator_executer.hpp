#ifndef OPERATOR_EXECUTER_HPP
#define OPERATOR_EXECUTER_HPP

#include "operator.hpp"
#include "code_generator.hpp"
#include <map>

using namespace std;

class operator_executer
{
public:
    operator_executer();

    // 在执行器中初始化一个元数据库
    operator_executer(shared_ptr<meta_data_set> meta_data_set_ptr)
    {
        assert(meta_data_set_ptr != NULL);
        this->meta_data_set_ptr = meta_data_set_ptr;
    }

    // operator执行器的运行
    void add_and_run(shared_ptr<basic_operator> current_operator);

    bool is_valid_of_inserted_operator(shared_ptr<basic_operator> current_operator);
    //获取上下文
    vector<shared_ptr<basic_operator>> get_operator_context(operator_stage_type type, int target_matrix_id)
    {
        return operator_history->read_operator_context_arr(type, target_matrix_id);
    }

    shared_ptr<operator_context> get_operator_context()
    {
        return operator_history;
    }
    //增减上下文
    void add_operator_context(shared_ptr<basic_operator> current_operator)
    {
        operator_history->operator_context_add(current_operator);
    }

    void pop_back_operator_context(operator_stage_type type, int target_matrix_id)
    {
        operator_history->pop_back_context(type, target_matrix_id);
    }

    vector<shared_ptr<transform_step_record_item>> get_transform_history()
    {
        return transform_history;
    }

    void push_back_tranform_step_in_history(vector<shared_ptr<transform_step_record_item>> current_operator_transform_step)
    {
        transform_history.insert(transform_history.end(), current_operator_transform_step.begin(), current_operator_transform_step.end());
    }

    void remove_transform_step_from_history(int index)
    {
        transform_history.erase(transform_history.begin() + index);
    }

    // 为对应的矩阵创建一个代码生成器，需要被implementing阶段的使用
    void add_code_generator_to_specific_sub_matrix(int sub_matrix_id);

    // 查看是否存在一个执行器
    bool if_code_generator_exist(int sub_matrix_id);

    // 获得特定子矩阵的执行器
    shared_ptr<code_generator> get_code_generator_of_spec_sub_matrix(int sub_matrix_id);

private:
    shared_ptr<operator_context> operator_history;
    vector<shared_ptr<transform_step_record_item>> transform_history;

    // 执行器中可能包含了多个代码生成器，每个子矩阵有一个代码生成器，使用一个键值对来存储每一个子矩阵对应的代码生成器
    map<int, shared_ptr<code_generator>> all_code_generator;

    // 使用一个指针来指向所有的元数据
    shared_ptr<meta_data_set> meta_data_set_ptr = NULL;
};

#endif