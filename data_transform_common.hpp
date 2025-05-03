#ifndef DATA_TRANSFORM_COMMON_HPP
#define DATA_TRANSFORM_COMMON_HPP

#include <vector>
#include "code_source_data.hpp"
#include "metadata_set.hpp"

using namespace std;

// 包含一些通用的工具函数，使用智能指针和通用数组的版本

// 获取一定范围的非零元的特定范围的行非零元的数量。是op_manager的get_nnz_of_each_row_in_spec_range的另外一个版本
// 要统计行非零元数量的行索引范围
vector<unsigned long> get_nnz_of_each_row_in_spec_range(shared_ptr<universal_array> row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end);
vector<unsigned int> get_nnz_of_each_row_in_spec_range_int(shared_ptr<universal_array> row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end);
unsigned int * get_nnz_of_each_row_in_spec_range_int(unsigned int * row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end);

vector<unsigned long> get_row_order_vec(shared_ptr<universal_array> row_index_arr, shared_ptr<universal_array> col_index_arr, unsigned long min_col_index, unsigned long max_col_index);
vector<unsigned long> get_nnz_of_each_col_in_spec_range(shared_ptr<universal_array> col_index_arr, unsigned long begin_col_bound, unsigned long end_col_bound, unsigned long global_coo_start, unsigned long global_coo_end);


// 通用数组的值拷贝
shared_ptr<universal_array> copy_universal_arr_by_value(shared_ptr<universal_array> source_array);

// 将一个子矩阵的表项原封不动值拷贝到另外一个子矩阵
void copy_item_in_metadata_set_by_value(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, string name, int source_sub_matrix_id, int dest_sub_matrix_id);

// 从metadata set中查看某一个层级的切块是不是是实际上的行切分
bool has_row_direction_blocking_in_specific_level(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, unsigned long sub_matrix_id);

// 判断某一级别的父块内的子块的大小是一样的
bool same_BMT_size_in_parent(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int sub_matrix_id);

// 查看两个pos之间的层次关系，能被另一个包含的层次称为更小的层次
bool former_pos_is_smaller_than_latter(POS_TYPE former_pos, POS_TYPE latter_pos);

bool padding_rate_valid_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int fixed_row_block, int sub_matrix_id);

bool padding_rate_valid_col_direction_with_multiple(shared_ptr<meta_data_set> meta_data_set_ptr, int fixed_col_block,  int sub_matrix_id);

bool padding_rate_valid_col_direction_with_max_size_in_parent(shared_ptr<meta_data_set> meta_data_set_ptr, int fixed_col_block,  POS_TYPE pos, int sub_matrix_id);

bool padding_rate_valid_empty_padding(shared_ptr<meta_data_set> meta_data_set_ptr, int sub_matrix_id);


vector<unsigned long> get_begin_nzs_of_child_after_balance_blocking_in_row_direction_in_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, shared_ptr<universal_array> parent_first_nzs_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval);
vector<unsigned long> get_begin_nzs_of_child_after_balance_blocking_in_row_direction_relative_to_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, shared_ptr<universal_array> parent_first_nzs_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval);

vector<unsigned long> get_begin_rows_of_child_after_balance_blocking_in_row_direction_in_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval, unsigned long row_num);
vector<unsigned long> get_begin_rows_of_child_after_balance_blocking_in_row_direction_relative_to_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval);

vector<unsigned long> get_begin_nzs_of_child_after_balance_blocking_in_row_direction(vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval);
vector<unsigned long> get_begin_rows_of_child_after_balance_blocking_in_row_direction(vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval, unsigned long row_num);

void read_mtx_as_csr_int(shared_ptr<meta_data_set> meta_data_set_ptr, vector<unsigned int> row_index_vec, unsigned long max_row_index);

string get_matrix_name(string filename);
#endif