#include "data_transform_common.hpp"
#include <vector>
#include <iostream>

using namespace std;

vector<unsigned long> get_nnz_of_each_row_in_spec_range(shared_ptr<universal_array> row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long begin_nz_bound, unsigned long end_nz_bound)
{
    // assert(row_index_arr != NULL);
    // assert(row_index_arr->check());
    // assert(row_index_arr->get_data_type() == UNSIGNED_LONG || row_index_arr->get_data_type() == UNSIGNED_INT || row_index_arr->get_data_type() == UNSIGNED_SHORT || row_index_arr->get_data_type() == UNSIGNED_CHAR);
    // assert(end_row_bound >= begin_row_bound && end_nz_bound >= begin_nz_bound);
    // assert(end_row_bound < row_index_arr->get_len());

    unsigned long total_row_num = end_row_bound - begin_row_bound + 1;

    // 申请一个数组存储每一行的非零元数量
    vector<unsigned long> nnz_of_each_row(total_row_num, 0);

    // 行号增序的检查，即便行顺序经过了一定的变化，行索引也是增序排列的，和原始索引的会产生一个映射，由其他数据负责
    unsigned long last_row_index = 0;

    // 遍历特定范围的非零元
    for (unsigned long cur_nz_index = begin_nz_bound; cur_nz_index <= end_nz_bound; cur_nz_index++)
    {
        unsigned long cur_row_index = row_index_arr->read_integer_from_arr(cur_nz_index);

        // 查看是不是增序
        // if (last_row_index > cur_row_index)
        // {
        //     cout << "last_row_index:" << last_row_index << ", "
        //          << "cur_row_index:" << cur_row_index << endl;
        //     assert(false);
        // }

        // assert(cur_row_index >= begin_row_bound && cur_row_index <= end_row_bound);

        // 计算相对行号
        unsigned long local_cur_row_index = cur_row_index - begin_row_bound;
        // assert(local_cur_row_index < total_row_num);

        // 相对位置的行非零元数量+1
        nnz_of_each_row[local_cur_row_index] = nnz_of_each_row[local_cur_row_index] + 1;
        last_row_index = cur_row_index;
    }

    // 每一行的元素数量
    return nnz_of_each_row;
}

vector<unsigned int> get_nnz_of_each_row_in_spec_range_int(shared_ptr<universal_array> row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long begin_nz_bound, unsigned long end_nz_bound)
{
    // assert(row_index_arr != NULL);
    // assert(row_index_arr->check());
    // assert(row_index_arr->get_data_type() == UNSIGNED_LONG || row_index_arr->get_data_type() == UNSIGNED_INT || row_index_arr->get_data_type() == UNSIGNED_SHORT || row_index_arr->get_data_type() == UNSIGNED_CHAR);
    // assert(end_row_bound >= begin_row_bound && end_nz_bound >= begin_nz_bound);
    // assert(end_row_bound < row_index_arr->get_len());
    assert(row_index_arr->get_data_type() == UNSIGNED_INT);
    unsigned int *row_indices = (unsigned int *)row_index_arr->get_arr_ptr(UNSIGNED_INT);
    unsigned long total_row_num = end_row_bound - begin_row_bound + 1;

    // 申请一个数组存储每一行的非零元数量
    vector<unsigned int> nnz_of_each_row(total_row_num, 0);
    unsigned long last_row_index = 0;

    // 遍历特定范围的非零元
    for (unsigned long cur_nz_index = begin_nz_bound; cur_nz_index <= end_nz_bound; cur_nz_index++)
    {
        unsigned long cur_row_index = row_indices[cur_nz_index];
        // 计算相对行号
        unsigned long local_cur_row_index = cur_row_index - begin_row_bound;

        // 相对位置的行非零元数量+1
        nnz_of_each_row[local_cur_row_index] = nnz_of_each_row[local_cur_row_index] + 1;
        last_row_index = cur_row_index;
    }

    // 每一行的元素数量
    return nnz_of_each_row;
}


unsigned int * get_nnz_of_each_row_in_spec_range_int(unsigned int * row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long begin_nz_bound, unsigned long end_nz_bound)
{
    // assert(row_index_arr != NULL);
    // assert(row_index_arr->check());
    // assert(row_index_arr->get_data_type() == UNSIGNED_LONG || row_index_arr->get_data_type() == UNSIGNED_INT || row_index_arr->get_data_type() == UNSIGNED_SHORT || row_index_arr->get_data_type() == UNSIGNED_CHAR);
    // assert(end_row_bound >= begin_row_bound && end_nz_bound >= begin_nz_bound);
    // assert(end_row_bound < row_index_arr->get_len());
    unsigned int *row_indices = row_index_arr;
    unsigned long total_row_num = end_row_bound - begin_row_bound + 1;

    // 申请一个数组存储每一行的非零元数量
    unsigned int * nnz_of_each_row;
    nnz_of_each_row = (unsigned int *)calloc(total_row_num, sizeof(unsigned int));

    unsigned long last_row_index = 0;

    // 遍历特定范围的非零元
    for (unsigned long cur_nz_index = begin_nz_bound; cur_nz_index <= end_nz_bound; cur_nz_index++)
    {
        unsigned long cur_row_index = row_indices[cur_nz_index];
        // 计算相对行号
        unsigned long local_cur_row_index = cur_row_index - begin_row_bound;

        // 相对位置的行非零元数量+1
        nnz_of_each_row[local_cur_row_index] = nnz_of_each_row[local_cur_row_index] + 1;
        last_row_index = cur_row_index;
    }

    // 每一行的元素数量
    return nnz_of_each_row;
}



vector<unsigned long> get_nnz_of_each_col_in_spec_range(shared_ptr<universal_array> col_index_arr, unsigned long begin_col_bound, unsigned long end_col_bound, unsigned long begin_nz_bound, unsigned long end_nz_bound)
{

    unsigned long total_col_num = end_col_bound - begin_col_bound + 1;

    vector<unsigned long> nnz_of_each_col(total_col_num, 0);

    // 遍历特定范围的非零元
    for (unsigned long cur_nz_index = begin_nz_bound; cur_nz_index <= end_nz_bound; cur_nz_index++)
    {
        unsigned long cur_col_index = col_index_arr->read_integer_from_arr(cur_nz_index);

        unsigned long local_cur_col_index = cur_col_index - begin_col_bound;
        // assert(local_cur_row_index < total_row_num);

        // 相对位置的行非零元数量+1
        nnz_of_each_col[local_cur_col_index] = nnz_of_each_col[local_cur_col_index] + 1;
    }

    // 每一行的元素数量
    return nnz_of_each_col;
}

vector<unsigned long> my_sort(vector<unsigned long> array)
{
    vector<unsigned long> index(array.size());
    for(unsigned long i=0; i < array.size(); i++)
    {
        index[i] = i;
    }
    for (unsigned long i = 0; i < array.size() - 1; i++)
    {

        for (unsigned long j = i + 1; j < array.size(); j++)
        {
            if (array[j] < array[i])
            {
                int tmp = array[j];
                array[j] = array[i];
                array[i] = tmp;

                tmp = index[i];
                index[i] = index[j];
                index[j] = tmp;
            }             
        }
    }
    return index;
}



vector<unsigned long> get_row_order_vec(shared_ptr<universal_array> row_index_arr, shared_ptr<universal_array> col_index_arr, unsigned long min_col_index, unsigned long max_col_index)
{

    vector<unsigned long> col_nz_number = get_nnz_of_each_col_in_spec_range(col_index_arr, min_col_index, max_col_index, 0, col_index_arr->get_len() - 1);

    vector<unsigned long> new_order_col = my_sort(col_nz_number);

    vector<vector<unsigned long>> bin_of_diff_col(max_col_index + 2);

    // 遍历所有的行非零元数量
    for (unsigned long cur_nz = 0; cur_nz < row_index_arr->get_len(); cur_nz++)
    {
        unsigned long nz_col = col_index_arr->read_integer_from_arr(cur_nz);
        unsigned long nz_row = row_index_arr->read_integer_from_arr(cur_nz);

        // 将行号放到对应的桶的末尾
        bin_of_diff_col[nz_col].push_back(nz_row);
    }

    set<unsigned long> row_index_done;

    vector<unsigned long> row_index_order_by_length_vec;

    for(unsigned long i = 0; i < new_order_col.size(); i++)
    {
        unsigned long col_ind = new_order_col[i];
        unsigned long next_col_ind = (i+1) < new_order_col.size()?  new_order_col[i + 1] : 0;
        for(unsigned long j = 0; j < bin_of_diff_col[col_ind].size(); j++)
        {
            unsigned long row_ind = bin_of_diff_col[col_ind][j];
            if (find(bin_of_diff_col[next_col_ind].begin(), bin_of_diff_col[next_col_ind].end(), row_ind) == bin_of_diff_col[next_col_ind].end())
            {
                if(row_index_done.count(row_ind) == 0)
                {
                    row_index_order_by_length_vec.push_back(row_ind);
                    row_index_done.insert(row_ind);
                }
            }
        }

        for(unsigned long j = 0; j < bin_of_diff_col[col_ind].size(); j++)
        {
            unsigned long row_ind = bin_of_diff_col[col_ind][j];
            if (find(bin_of_diff_col[next_col_ind].begin(), bin_of_diff_col[next_col_ind].end(), row_ind) != bin_of_diff_col[next_col_ind].end())
            {
                if (row_index_done.count(row_ind) == 0)
                {
                    row_index_order_by_length_vec.push_back(row_ind);
                    row_index_done.insert(row_ind);
                }
            }
        }
    }

    return row_index_order_by_length_vec;

}

shared_ptr<universal_array> copy_universal_arr_by_value(shared_ptr<universal_array> source_array)
{
    // 不是空的，并且检查
    assert(source_array != NULL);
    assert(source_array->check() == true);
    assert(source_array->get_len() > 0);
    vector<double> double_new_content_vec;
    vector<bool> bool_new_content_vec;
    vector<unsigned long> unsigned_long_new_content_vec;

    // 分成bool、整型、浮点型三种情况
    if (source_array->get_data_type() == DOUBLE || source_array->get_data_type() == FLOAT)
    {
        data_type type = source_array->get_data_type();

        // 遍历矩阵中所有内容
        for (int i = 0; i < source_array->get_len(); i++)
        {
            double cur_content = source_array->read_float_from_arr(i);
            double_new_content_vec.push_back(cur_content);
        }

        // 创建一个新的通用数组
        shared_ptr<universal_array> dest_array(new universal_array(((void *)&(double_new_content_vec[0])), double_new_content_vec.size(), DOUBLE));

        if (type == FLOAT)
        {
            dest_array->compress_float_precise();
            assert(dest_array->get_data_type() == FLOAT);
        }

        return dest_array;
    }
    else if (source_array->get_data_type() == BOOL)
    {
        // 遍历所有内容
        // for (int i = 0; i < source_array->get_len(); i++)
        // {
        //     // bool cur_content = source_array->read_integer_from_arr(i);
        //     bool_new_content_vec.push_back(true);
        // }

        // // vector<bool> A;
        // // A.push_back();

        // // 创建一个新的数组
        // shared_ptr<universal_array> dest_array(new universal_array(((void *)(&bool_new_content_vec[0])), bool_new_content_vec.size(), BOOL));

        // return dest_array;
        // bool类型存入vector，再用下标运算符取出来的是reference类的拷贝，而不是值的引用，所以本质上不支持这个
        cout << "copy_universal_arr_by_value:bool type is not supported" << endl;
        assert(false);
    }
    else if (source_array->get_data_type() == UNSIGNED_LONG)
    {
        // 遍历所有内容
        for (int i = 0; i < source_array->get_len(); i++)
        {
            unsigned long cur_content = source_array->read_integer_from_arr(i);
            unsigned_long_new_content_vec.push_back(cur_content);
        }

        // 创建新的数组
        shared_ptr<universal_array> dest_array(new universal_array(((void *)(&unsigned_long_new_content_vec[0])), unsigned_long_new_content_vec.size(), UNSIGNED_LONG));

        return dest_array;
    }

    // 这里是必然的错误
    assert(false);
    return NULL;
}

void copy_item_in_metadata_set_by_value(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, string name, int source_sub_matrix_id, int dest_sub_matrix_id)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check() == true);
    assert(check_pos_type(pos) == true);
    assert(source_sub_matrix_id >= -1);
    assert(dest_sub_matrix_id >= -1);

    // 首先查看源表项存在，并且目标不存在
    assert(meta_data_set_ptr->is_exist(pos, name, source_sub_matrix_id) == true);
    assert(meta_data_set_ptr->is_exist(pos, name, dest_sub_matrix_id) == false);

    // 将内容读出并创造新的表项
    shared_ptr<universal_array> source_arr_ptr = meta_data_set_ptr->get_element(pos, name, source_sub_matrix_id)->get_metadata_arr();
    // 对这一表项执行值拷贝
    shared_ptr<universal_array> dest_arr_ptr = copy_universal_arr_by_value(source_arr_ptr);

    // 初始化一个表项，注意常量和数组类型要保持一致
    shared_ptr<meta_data_item> dest_item_ptr(new meta_data_item(dest_arr_ptr, pos, name, dest_sub_matrix_id));
    dest_item_ptr->metadata_type = meta_data_set_ptr->get_element(pos, name, source_sub_matrix_id)->metadata_type;

    if (dest_item_ptr->metadata_type == CON_META_TYPE)
    {
        assert(dest_arr_ptr->get_len() == 1);
    }

    // 将表项插入元数据表中
    meta_data_set_ptr->add_element(dest_item_ptr);
}

bool has_row_direction_blocking_in_specific_level(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, unsigned long sub_matrix_id)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check() == true);
    assert(pos == THREAD_META || pos == WARP_META || pos == TBLOCK_META);

    //存在without ending 则为列切分
    if (meta_data_set_ptr->is_exist(pos, "first_row_indices_without_ending", sub_matrix_id) == true)
    {
        return false;
    }
    // TBLOCK和WARP、THREAD是相同的处理
    else
    {
        // 首先查看对应的元素是不是存在，CSR-like的行偏移量
        assert(meta_data_set_ptr->is_exist(pos, "first_row_indices", sub_matrix_id));
        // 将对应的行索引取出
        shared_ptr<universal_array> first_row_indices_ptr = meta_data_set_ptr->get_element(pos, "first_row_indices", sub_matrix_id)->get_metadata_arr();

        // 遍历所有的元素，如果出现两个相同的行索引，就返回false
        unsigned long prev_first_row_index = first_row_indices_ptr->read_integer_from_arr(0);
        // 查看重复的次数
        unsigned long repeat_num = 1;

        for (unsigned long i = 1; i < first_row_indices_ptr->get_len(); i++)
        {
            if (first_row_indices_ptr->read_integer_from_arr(i) == prev_first_row_index)
            {
                repeat_num++;
            }
            else
            {
                repeat_num = 1;
            }

            if (repeat_num == 2)
            {
                return false;
            }

            // 记录之前的行索引
            prev_first_row_index = first_row_indices_ptr->read_integer_from_arr(i);
        }
    }

    return true;
}

bool same_BMT_size_in_parent(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int sub_matrix_id)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(sub_matrix_id >= 0);
    assert(pos == GLOBAL_META || pos == TBLOCK_META || pos == WARP_META);

    // 查看当前父块非零元偏移，和BMT的非零元偏移，都是绝对偏移，从而推测出每个父块中BMT的大小
    assert(meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", sub_matrix_id));
    if (pos == TBLOCK_META || pos == WARP_META)
    {
        assert(meta_data_set_ptr->is_exist(pos, "first_nz_indices", sub_matrix_id));
    }

    // 读出来数组
    shared_ptr<universal_array> first_nz_indices_of_BMT_ptr = meta_data_set_ptr->get_element(THREAD_META, "first_nz_indices", sub_matrix_id)->get_metadata_arr();
    shared_ptr<universal_array> first_nz_indices_of_parent_ptr = NULL;
    if (pos == TBLOCK_META || pos == WARP_META)
    {
        first_nz_indices_of_parent_ptr = meta_data_set_ptr->get_element(pos, "first_nz_indices", sub_matrix_id)->get_metadata_arr();
    }

    // 遍历BMT的索引，会不断自增
    unsigned long cur_BMT_id = 0;

    if (pos == GLOBAL_META)
    {
        // 当前的父块的BMT的大小，用来比较是不是一样的
        unsigned long cur_BMT_size_of_parent = 0;
        bool go_through_the_first_BMT = false;

        // 遍历当前父块的所有的BMT
        while (cur_BMT_id < first_nz_indices_of_BMT_ptr->get_len() - 1)
        {
            // 当前BMT的偏移量
            unsigned long cur_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);

            // 下一个BMT的偏移量
            if (cur_BMT_id + 1 >= first_nz_indices_of_BMT_ptr->get_len())
            {
                cout << "get_BMT_size_of_each_parent::run(): cur_BMT_id + 1:" << cur_BMT_id + 1 << ", first_nz_indices_of_BMT_ptr->get_len():" << first_nz_indices_of_BMT_ptr->get_len() << endl;
                assert(false);
            }

            unsigned long next_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id + 1);
            // 当前BMT的大小
            assert(next_BMT_first_nz_id > cur_BMT_first_nz_id);
            unsigned long BMT_size = next_BMT_first_nz_id - cur_BMT_first_nz_id;

            if (go_through_the_first_BMT == false)
            {
                cur_BMT_size_of_parent = BMT_size;
                go_through_the_first_BMT = true;
            }
            else
            {
                // 这里做一个检查，保证当前的BMT大小和之前的都是一样的
                if (BMT_size != cur_BMT_size_of_parent)
                {
                    return false;
                }
            }

            // 索引自增
            cur_BMT_id++;
            // 为了退出条件正确要查看当前的行偏移量
            cur_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);
        }
        // 检查，如果当前父块不是空的，那内部的BMT的
        if (go_through_the_first_BMT == false)
        {
            assert(cur_BMT_size_of_parent == 0);
        }
        else
        {
            assert(cur_BMT_size_of_parent > 0);
        }
    }
    else
    {
        // 遍历所有的父块的索引，查看内部所有的BMT的非零元索引
        for (unsigned long parent_blk_id = 0; parent_blk_id < first_nz_indices_of_parent_ptr->get_len() - 1; parent_blk_id++)
        {
            // 当前父块和下一个父块的非零元索引
            unsigned long cur_parent_blk_first_nz_id = first_nz_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id);
            unsigned long next_parent_blk_first_nz_id = first_nz_indices_of_parent_ptr->read_integer_from_arr(parent_blk_id + 1);

            // 当前BMT的非零元偏移量
            assert(cur_BMT_id < first_nz_indices_of_BMT_ptr->get_len());
            unsigned long cur_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);

            // 在一开始的时候，BMT的非零元偏移量和父块的非零元偏移量是相同的
            if (cur_BMT_first_nz_id != cur_parent_blk_first_nz_id)
            {
                cout << "get_BMT_size_of_each_parent::run(): cur_BMT_first_nz_id:" << cur_BMT_first_nz_id << ", cur_parent_blk_first_nz_id:" << cur_parent_blk_first_nz_id << endl;
                assert(false);
            }

            // 当前的父块的BMT的大小，用来比较是不是一样的，并且最终用来记录BMT的大小
            unsigned long cur_BMT_size_of_parent = 0;
            bool go_through_the_first_BMT = false;

            // 遍历当前父块的所有的BMT
            while (cur_BMT_first_nz_id < next_parent_blk_first_nz_id)
            {
                // 当前BMT的偏移量
                assert(cur_BMT_id < first_nz_indices_of_BMT_ptr->get_len());
                cur_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);

                // 下一个BMT的偏移量

                if (cur_BMT_id + 1 >= first_nz_indices_of_BMT_ptr->get_len())
                {
                    cout << "get_BMT_size_of_each_parent::run(): cur_BMT_id + 1:" << cur_BMT_id + 1 << ", first_nz_indices_of_BMT_ptr->get_len():" << first_nz_indices_of_BMT_ptr->get_len() << endl;
                    assert(false);
                }

                unsigned long next_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id + 1);

                // 当前BMT的大小
                assert(next_BMT_first_nz_id > cur_BMT_first_nz_id);
                unsigned long BMT_size = next_BMT_first_nz_id - cur_BMT_first_nz_id;

                if (go_through_the_first_BMT == false)
                {
                    cur_BMT_size_of_parent = BMT_size;
                    go_through_the_first_BMT = true;
                }
                else
                {
                    // 这里做一个检查，保证当前的BMT大小和之前的都是一样的
                    if (BMT_size != cur_BMT_size_of_parent)
                    {
                        return false;
                    }
                }

                // 索引自增
                cur_BMT_id++;

                // 为了退出条件正确要查看当前的行偏移量
                cur_BMT_first_nz_id = first_nz_indices_of_BMT_ptr->read_integer_from_arr(cur_BMT_id);
            }

            // 检查，如果当前父块不是空的，那内部的BMT的
            if (go_through_the_first_BMT == false)
            {
                assert(cur_BMT_size_of_parent == 0);
            }
            else
            {
                assert(cur_BMT_size_of_parent > 0);
            }
        }
    }

    // 检查，BMT已经全部遍历完
    assert(cur_BMT_id == first_nz_indices_of_BMT_ptr->get_len() - 1);
    return true;
}

bool former_pos_is_smaller_than_latter(POS_TYPE former_pos, POS_TYPE latter_pos)
{
    // 先检查两个输入
    assert(check_pos_type(former_pos) == true);
    assert(check_pos_type(latter_pos) == true);
    // 不能是NONE类型
    assert(former_pos == GLOBAL_META || former_pos == TBLOCK_META || former_pos == WARP_META || former_pos == THREAD_META);
    assert(latter_pos == GLOBAL_META || latter_pos == TBLOCK_META || latter_pos == WARP_META || latter_pos == THREAD_META);

    // 将POS转化为数字，越外层的优先级越高
    int former_priority = priority_of_pos_type(former_pos);
    int latter_priority = priority_of_pos_type(latter_pos);

    // 比较数字，查看前者的数字是不是小于后者的数字
    return former_priority < latter_priority;
}

bool padding_rate_valid_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int fixed_row_block, int sub_matrix_id)
{
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", sub_matrix_id));

    unsigned long begin_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 行索引

    shared_ptr<universal_array> row_indices_ptr = meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", sub_matrix_id)->get_metadata_arr();

    unsigned long real_end_row_index = begin_row_index + row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }

    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;

    if (row_num_of_sub_matrix % fixed_row_block != 0)
    {
        // 目标行数量
        unsigned long new_row_number = (row_num_of_sub_matrix / fixed_row_block + 1) * fixed_row_block;
        // 计算要padding的行（非零元）数量
        unsigned long added_row_number = new_row_number - row_num_of_sub_matrix;

        nnz_after_padding = nnz_after_padding + added_row_number;

        // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "modify_col_indices_by_row_pad_in_sub_matrix::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            return false;
        }
    }
    return true;
}

bool padding_rate_valid_empty_padding(shared_ptr<meta_data_set> meta_data_set_ptr, int sub_matrix_id)
{

    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", sub_matrix_id));

    unsigned long begin_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 行索引
    shared_ptr<universal_array> row_indices_ptr = meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", sub_matrix_id)->get_metadata_arr();

    unsigned long real_end_row_index = begin_row_index + row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }
    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(row_indices_ptr, 0, row_num_of_sub_matrix - 1, 0, row_indices_ptr->get_len() - 1);

    if (find(nnz_of_each_row.begin(), nnz_of_each_row.end(), 0) != nnz_of_each_row.end())
    {
        // 目标行数量
        unsigned long new_row_number = row_num_of_sub_matrix + count(nnz_of_each_row.begin(), nnz_of_each_row.end(), 0);
        // 计算要padding的行（非零元）数量
        unsigned long added_row_number = new_row_number - row_num_of_sub_matrix;

        nnz_after_padding = nnz_after_padding + added_row_number;

        // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
        if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "modify_col_indices_by_row_pad_in_sub_matrix::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            return false;
        }
    }
    return true;
}

bool padding_rate_valid_col_direction_with_multiple(shared_ptr<meta_data_set> meta_data_set_ptr, int fixed_col_block, int sub_matrix_id)
{

    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", sub_matrix_id));

    unsigned long begin_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 行索引
    shared_ptr<universal_array> row_indices_ptr = meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", sub_matrix_id)->get_metadata_arr();

    unsigned long real_end_row_index = begin_row_index + row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }
    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(row_indices_ptr, 0, row_num_of_sub_matrix - 1, 0, row_indices_ptr->get_len() - 1);

    // 遍历每一个桶，根据当前桶的大小执行padding，padding到multiple_of_each_row_size的整数倍
    for (unsigned long i = 0; i < row_num_of_sub_matrix; i++)
    {
        // 当前行长度
        unsigned long cur_row_size = nnz_of_each_row[i];
        // 目标行长度
        if (cur_row_size % fixed_col_block != 0)
        {
            unsigned long target_row_size = (cur_row_size / fixed_col_block + 1) * fixed_col_block;

            unsigned long added_row_nnz = target_row_size - cur_row_size;

            nnz_after_padding = nnz_after_padding + added_row_nnz;

            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
            {
                cout << "modify_row_indices_by_col_pad_in_sub_matrix::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                return false;
            }
        }
    }
    return true;
}

bool padding_rate_valid_col_direction_with_max_size_in_parent(shared_ptr<meta_data_set> meta_data_set_ptr, int fixed_col_block, POS_TYPE pos, int sub_matrix_id)
{
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", sub_matrix_id));
    assert(meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", sub_matrix_id));

    unsigned long begin_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", sub_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    // 行索引
    shared_ptr<universal_array> row_indices_ptr = meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", sub_matrix_id)->get_metadata_arr();

    unsigned long real_end_row_index = begin_row_index + row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    if (end_row_index < real_end_row_index)
    {
        end_row_index = real_end_row_index;
    }

    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小
    unsigned long original_nnz = row_indices_ptr->get_len();
    unsigned long nnz_after_padding = original_nnz;
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(row_indices_ptr, 0, row_num_of_sub_matrix - 1, 0, row_indices_ptr->get_len() - 1);

    if (pos == GLOBAL_META)
    {
        // 查看最长的行
        unsigned long max_row_length = *max_element(nnz_of_each_row.begin(), nnz_of_each_row.end());

        // 遍历所有的行
        for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
        {
            // 当前行的非零元数量
            unsigned long cur_row_size = nnz_of_each_row[i];

            // 查看是不是要执行padding
            if (max_row_length != 0 && cur_row_size != 0)
            {
                unsigned long target_row_length = max_row_length;

                // 查看要增加的非零元数量
                unsigned long added_row_nz_num = target_row_length - cur_row_size;

                nnz_after_padding = nnz_after_padding + added_row_nz_num;

                // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
                if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                {
                    cout << "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                    return false;
                }
            }
        }
    }
    else if (pos == TBLOCK_META || pos == WARP_META)
    {
        assert(meta_data_set_ptr->is_exist(pos, "first_row_indices", sub_matrix_id));
        shared_ptr<universal_array> first_row_indices_of_parent_ptr = meta_data_set_ptr->get_element(pos, "first_row_indices", sub_matrix_id)->get_metadata_arr();

        // 遍历所有父块
        for (unsigned long i = 0; i < first_row_indices_of_parent_ptr->get_len() - 1; i++)
        {
            // 获得当前父块的行索引范围
            unsigned long first_row_index_of_cur_parent = first_row_indices_of_parent_ptr->read_integer_from_arr(i);
            unsigned long first_row_index_of_next_parent = first_row_indices_of_parent_ptr->read_integer_from_arr(i + 1);

            // 记录当前父块中最大的行长度
            unsigned long max_row_size_of_this_parent = 0;

            // 遍历当前区间的行非零元，找到最大的行长度
            for (unsigned long row_index = first_row_index_of_cur_parent; row_index < first_row_index_of_next_parent; row_index++)
            {
                unsigned long cur_row_size = nnz_of_each_row[row_index];
                if (max_row_size_of_this_parent < cur_row_size)
                {
                    max_row_size_of_this_parent = cur_row_size;
                }
            }

            for (unsigned long row_index = first_row_index_of_cur_parent; row_index < first_row_index_of_next_parent; row_index++)
            {
                unsigned long cur_row_size = nnz_of_each_row[row_index];
                if (max_row_size_of_this_parent != 0 && cur_row_size != 0)
                {
                    unsigned long target_row_length = max_row_size_of_this_parent;
                    unsigned long added_row_nnz = target_row_length - cur_row_size;
                    nnz_after_padding = nnz_after_padding + added_row_nnz;
                    if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
                    {
                        cout << "modify_row_indices_by_col_pad_parent_blk_to_max_row_size::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

vector<unsigned long> get_begin_nzs_of_child_after_balance_blocking_in_row_direction_in_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, shared_ptr<universal_array> parent_first_nzs_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval)
{
    vector<unsigned long> begin_nz_vec;

    // 遍历所有父块，父块的数量为parent_first_row_indices_ptr长度-1
    assert(parent_first_row_indices_ptr->get_len() == parent_first_nzs_ptr->get_len());
    for (unsigned long j = 0; j < parent_first_row_indices_ptr->get_len() - 1; j++)
    {
        // 当前BMTB的行偏移
        unsigned long parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j);
        unsigned long next_parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j + 1);

        // 当前BMTB非零元偏移量
        unsigned long parent_first_nz = parent_first_nzs_ptr->read_integer_from_arr(j);
        unsigned long next_parent_first_nz = parent_first_nzs_ptr->read_integer_from_arr(j + 1);

        unsigned int nz_count = 0;
        unsigned long cur_nz_in_block = 0;
        begin_nz_vec.push_back(parent_first_nz);

        // 遍历所有的行
        for (unsigned long i = parent_first_row_index; i < next_parent_first_row_index; i++)
        {
            unsigned long cur_row_size = nnz_of_each_row[i];

            nz_count += cur_row_size;
            cur_nz_in_block += cur_row_size;
            // 正好与父块边界对齐不必计算，下一个父块会插入首个非零元
            if (nz_count >= nnz_per_interval && i != (next_parent_first_row_index - 1))
            {
                begin_nz_vec.push_back(parent_first_nz + cur_nz_in_block);
                nz_count = 0;
            }
        }
    }

    begin_nz_vec.push_back(parent_first_nzs_ptr->read_integer_from_arr(parent_first_nzs_ptr->get_len() - 1));
    return begin_nz_vec;
}

vector<unsigned long> get_begin_nzs_of_child_after_balance_blocking_in_row_direction_relative_to_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, shared_ptr<universal_array> parent_first_nzs_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval)
{
    vector<unsigned long> begin_nz_vec;
    // 遍历所有父块
    for (unsigned long j = 0; j < parent_first_row_indices_ptr->get_len() - 1; j++)
    {
        // 当前父块的行偏移
        unsigned long parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j);
        unsigned long next_parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j + 1);

        // 当前父块非零元偏移量
        unsigned long parent_first_nz = parent_first_nzs_ptr->read_integer_from_arr(j);
        unsigned long next_parent_first_nz = parent_first_nzs_ptr->read_integer_from_arr(j + 1);

        unsigned int nz_count = 0;
        unsigned long cur_nz_in_block = 0;
        begin_nz_vec.push_back(0);

        // 遍历所有的行，按照非零元数量分块，包含零行
        for (unsigned long i = parent_first_row_index; i < next_parent_first_row_index; i++)
        {
            unsigned long cur_row_size = nnz_of_each_row[i];

            nz_count += cur_row_size;
            cur_nz_in_block += cur_row_size;
            //正好与父块边界对齐不必计算，下一个父块会插入首个非零元
            if (nz_count >= nnz_per_interval && i != (next_parent_first_row_index - 1))
            {
                begin_nz_vec.push_back(cur_nz_in_block);
                nz_count = 0;
            }
        }
    }

    return begin_nz_vec;
}

vector<unsigned long> get_begin_rows_of_child_after_balance_blocking_in_row_direction_in_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval, unsigned long row_num)
{
    vector<unsigned long> begin_row_vec;
    // 遍历所有父块
    for (unsigned long j = 0; j < parent_first_row_indices_ptr->get_len() - 1; j++)
    {
        // 当前父块的行偏移
        unsigned long parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j);
        unsigned long next_parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j + 1);
        begin_row_vec.push_back(parent_first_row_index);
        unsigned nz_count = 0;

        // 遍历所有的行，按照非零元数量分块，包含零行
        for (unsigned long i = parent_first_row_index; i < next_parent_first_row_index; i++)
        {

            unsigned long cur_row_nnz = nnz_of_each_row[i];

            nz_count = nz_count + cur_row_nnz;
            //正好与父块边界对齐不必计算，下一个父块会插入
            if (nz_count >= nnz_per_interval && i != (next_parent_first_row_index - 1))
            {
                begin_row_vec.push_back(i + 1);
                nz_count = 0;
            }
        }
    }
    begin_row_vec.push_back(row_num);

    return begin_row_vec;
}

vector<unsigned long> get_begin_rows_of_child_after_balance_blocking_in_row_direction_relative_to_parent(shared_ptr<universal_array> parent_first_row_indices_ptr, vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval)
{
    vector<unsigned long> begin_row_vec;
    // 遍历所有父块
    for (unsigned long j = 0; j < parent_first_row_indices_ptr->get_len() - 1; j++)
    {
        // 当前父块的行偏移
        unsigned long parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j);
        unsigned long next_parent_first_row_index = parent_first_row_indices_ptr->read_integer_from_arr(j + 1);
        begin_row_vec.push_back(0);
        unsigned nz_count = 0;

        // 遍历父块内部所有的行，按照非零元数量分块，包含零行
        for (unsigned long i = parent_first_row_index; i < next_parent_first_row_index; i++)
        {

            unsigned long cur_row_nnz = nnz_of_each_row[i];

            nz_count = nz_count + cur_row_nnz;
            //正好与父块边界对齐不必计算，下一个父块会插入
            if (nz_count >= nnz_per_interval && i != (next_parent_first_row_index - 1))
            {
                begin_row_vec.push_back(i + 1 - parent_first_row_index);
                nz_count = 0;
            }
        }
    }

    return begin_row_vec;
}

vector<unsigned long> get_begin_nzs_of_child_after_balance_blocking_in_row_direction(vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval)
{
    vector<unsigned long> begin_nz_vec;
    begin_nz_vec.push_back(0);
    unsigned int nz_count = 0;
    unsigned int cur_nz_total = 0;
    // 遍历所有的行，按照非零元数量分块，包含零行
    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        unsigned long cur_row_size = nnz_of_each_row[i];

        nz_count += cur_row_size;
        cur_nz_total += cur_row_size;
        if (nz_count >= nnz_per_interval)
        {
            begin_nz_vec.push_back(cur_nz_total);
            nz_count = 0;
        }
    }
    if (nz_count != 0)
    {
        begin_nz_vec.push_back(cur_nz_total);
    }
    return begin_nz_vec;
}

vector<unsigned long> get_begin_rows_of_child_after_balance_blocking_in_row_direction(vector<unsigned long> nnz_of_each_row, unsigned long nnz_per_interval, unsigned long row_num)
{
    vector<unsigned long> begin_row_vec;
    begin_row_vec.push_back(0);

    unsigned long nz_count = 0;
    // 遍历所有的行，按照非零元数量分块，包含零行
    for (unsigned long i = 0; i < row_num; i++)
    {
        unsigned long cur_row_nnz = nnz_of_each_row[i];
        nz_count = nz_count + cur_row_nnz;

        if (nz_count >= nnz_per_interval)
        {
            begin_row_vec.push_back(i + 1);
            nz_count = 0;
        }
    }

    // 最后一个切分点强制是整个子矩阵的行数量
    assert(begin_row_vec[begin_row_vec.size() - 1] <= row_num);
    
    if (begin_row_vec[begin_row_vec.size() - 1] < row_num)
    {
        assert(nz_count != 0);
        begin_row_vec.push_back(row_num);    
    }

    return begin_row_vec;
}

void read_mtx_as_csr_int(shared_ptr<meta_data_set> meta_data_set_ptr, vector<unsigned int> row_index_vec, unsigned long max_row_index)
{
    vector<unsigned int> nnz_of_each_row(max_row_index + 1, 0);
    for (unsigned long cur_nz_index = 0; cur_nz_index < row_index_vec.size(); cur_nz_index++)
    {
        unsigned long cur_row_index = row_index_vec[cur_nz_index];

        nnz_of_each_row[cur_row_index] = nnz_of_each_row[cur_row_index] + 1;
    }

    vector<unsigned int> CSR_vec;
    CSR_vec.push_back(0);
    int nz_count = 0;

    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        unsigned int cur_row_size = nnz_of_each_row[i];
        nz_count += cur_row_size;
        CSR_vec.push_back(nz_count); 
    }

    // 将新的内容放到metadata set中
    shared_ptr<universal_array> CSR_ptr(new universal_array(&(CSR_vec[0]), CSR_vec.size(), UNSIGNED_INT));
    shared_ptr<meta_data_item> CSR_item(new meta_data_item(CSR_ptr, GLOBAL_META,  "CSR_row_indices", 0));
    meta_data_set_ptr->add_element(CSR_item);    
}

string get_matrix_name(string filename)
{
    string matrix_name;
    int index = filename.find_last_of('/');
    for (int i = index + 1; i < filename.size(); i++)
    {
        if (filename[i] != '.')
        {
            matrix_name += filename[i];
        }
        else
        {
            break;
        }
    }
    return matrix_name;
}