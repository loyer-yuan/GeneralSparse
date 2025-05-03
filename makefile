all: main

CXX := g++
LD := ${CXX}

# Flags to enable link-time optimization and GDB
LTO := -fno-lto
ENABLE_DGB :=

BUILDER_HOME := .

INC	:= -I ${BUILDER_HOME}

# DEBUG := -DNDEBUG
CPPFLAGS := ${ENABLE_DGB} ${LTO} -g -O3 ${DEBUG} -std=c++11 ${INC} \
	-Wno-unused-result -Wno-unused-value -Wno-unused-function \
	# -Winline

LDFLAGS := ${ENABLE_DGB} ${LTO} -pthread

src :=${BUILDER_HOME}/config.o ${BUILDER_HOME}/op_manager.o ${BUILDER_HOME}/struct.o ${BUILDER_HOME}/arr_optimization.o ${BUILDER_HOME}/empty_op.o \
${BUILDER_HOME}/dataset_builder.o ${BUILDER_HOME}/code_source_data.o ${BUILDER_HOME}/metadata_set.o ${BUILDER_HOME}/data_transform_common.o ${BUILDER_HOME}/data_transform_graph.o ${BUILDER_HOME}/transform_step/get_row_order_by_length.o \
${BUILDER_HOME}/data_transform_step.o ${BUILDER_HOME}/transform_step/reorder_row_by_index.o ${BUILDER_HOME}/transform_step/reorder_col_by_index.o \
${BUILDER_HOME}/operator/sort_operator.o ${BUILDER_HOME}/operator.o ${BUILDER_HOME}/transform_step/reorder_val_by_index.o ${BUILDER_HOME}/transform_step/fixed_div_col_indices_by_corr_row_indices.o \
${BUILDER_HOME}/transform_step/fixed_div_vals_by_corr_row_indices.o ${BUILDER_HOME}/transform_step/fixed_div_row_indices.o ${BUILDER_HOME}/transform_step/modify_row_start_boundary_after_fixed_div_in_row_direction.o \
${BUILDER_HOME}/transform_step/modify_row_end_boundary_after_fixed_div_in_row_direction.o ${BUILDER_HOME}/transform_step/modify_col_start_boundary_after_fixed_div_in_row_direction.o \
${BUILDER_HOME}/transform_step/modify_col_end_boundary_after_fixed_div_in_row_direction.o ${BUILDER_HOME}/operator/fixed_interval_row_matrix_div_operator.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction.o \
${BUILDER_HOME}/transform_step/remove_empty_row_in_end_of_sub_matrix.o ${BUILDER_HOME}/transform_step/modify_row_indices_by_row_pad_in_sub_matrix.o \
${BUILDER_HOME}/transform_step/modify_col_indices_by_row_pad_in_sub_matrix.o ${BUILDER_HOME}/transform_step/modify_vals_by_row_pad_in_sub_matrix.o ${BUILDER_HOME}/operator/fixed_interval_row_direction_tblock_blocking_operator.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB.o \
${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB.o \
${BUILDER_HOME}/operator/fixed_interval_row_direction_warp_blocking_operator.o ${BUILDER_HOME}/transform_step/modify_row_indices_by_col_pad_in_sub_matrix.o ${BUILDER_HOME}/transform_step/modify_col_indices_by_col_pad_in_sub_matrix.o \
${BUILDER_HOME}/transform_step/modify_vals_by_col_pad_in_sub_matrix.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMTB.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction.o \
${BUILDER_HOME}/transform_step/get_begin_BMWs_of_BMTB_after_blocking_in_row_direction.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents.o \
${BUILDER_HOME}/transform_step/get_BMT_size_of_each_parent.o ${BUILDER_HOME}/transform_step/get_begin_BMTs_of_specific_parent_after_blocking.o ${BUILDER_HOME}/transform_step/modify_col_indices_by_col_pad_parent_blk_to_max_row_size.o \
${BUILDER_HOME}/transform_step/modify_vals_by_col_pad_parent_blk_to_max_row_size.o ${BUILDER_HOME}/transform_step/modify_row_indices_by_col_pad_parent_blk_to_max_row_size.o ${BUILDER_HOME}/kernel_generator.o ${BUILDER_HOME}/kernel_token/math_expr_token.o \
${BUILDER_HOME}/kernel_token/var_name_token.o ${BUILDER_HOME}/kernel_token/arr_access_token.o ${BUILDER_HOME}/kernel_token/data_type_token.o ${BUILDER_HOME}/kernel_token/var_init_token.o ${BUILDER_HOME}/kernel_token/shared_mem_init_token.o \
${BUILDER_HOME}/kernel_token/shared_mem_write_token.o ${BUILDER_HOME}/kernel_token/metadata_get_token.o ${BUILDER_HOME}/kernel_token/reduction_basic_token.o ${BUILDER_HOME}/transform_step/modify_col_indices_by_empty_pad_in_submatrix.o \
${BUILDER_HOME}/transform_step/modify_vals_by_empty_pad_in_submatrix.o ${BUILDER_HOME}/transform_step/modify_row_indices_by_empty_pad_in_submatrix.o ${BUILDER_HOME}/operator/empty_row_pad_operator.o ${BUILDER_HOME}/transform_step/remove_item_of_metadata.o \
${BUILDER_HOME}/operator/fixed_interval_col_direction_thread_blocking_operator.o ${BUILDER_HOME}/transform_step/modify_col_indices_by_interlance_storage.o ${BUILDER_HOME}/transform_step/modify_row_indices_by_interlance_storage.o \
${BUILDER_HOME}/transform_step/modify_vals_by_interlance_storage.o ${BUILDER_HOME}/operator/interlance_storage_operator.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMW.o \
${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW.o ${BUILDER_HOME}/operator/fixed_interval_row_direction_thread_blocking_operator.o ${BUILDER_HOME}/kernel_token/for_basic_token.o ${BUILDER_HOME}/kernel_token/for_token.o ${BUILDER_HOME}/kernel_token/shared_mem_broadcast_token.o \
${BUILDER_HOME}/transform_step/get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction.o ${BUILDER_HOME}/operator/fixed_interval_col_direction_warp_blocking_operator.o ${BUILDER_HOME}/transform_step/get_begin_BMWs_of_BMTB_after_blocking.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB.o \
${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_col_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction.o \
${BUILDER_HOME}/transform_step/get_BMW_size_of_each_parent.o ${BUILDER_HOME}/kernel_token/var_assign_token.o ${BUILDER_HOME}/kernel_token/metadata_set_get_token.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMTB_after_fixed_blocking_in_col_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction.o ${BUILDER_HOME}/transform_step/get_BMTB_size.o \
${BUILDER_HOME}/operator/fixed_interval_col_direction_tblock_blocking_operator.o ${BUILDER_HOME}/operator_executer.o ${BUILDER_HOME}/operator/row_nz_matrix_div_operator.o ${BUILDER_HOME}/transform_step/div_col_indices_by_row_nnz.o ${BUILDER_HOME}/transform_step/div_row_indices_by_row_nnz.o ${BUILDER_HOME}/transform_step/div_val_indices_by_row_nnz.o \
${BUILDER_HOME}/transform_step/modify_col_end_boundary_after_div_according_to_row_nz.o ${BUILDER_HOME}/transform_step/modify_col_start_boundary_after_div_according_to_row_nz.o ${BUILDER_HOME}/transform_step/modify_row_end_boundary_after_div_according_to_row_nz.o ${BUILDER_HOME}/transform_step/modify_row_start_boundary_after_div_according_to_row_nz.o \
${BUILDER_HOME}/operator/balanced_interval_row_direction_thread_blocking_operator.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction.o \
${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_in_BMW.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMW.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_in_BMW.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMW.o \
${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMTB.o ${BUILDER_HOME}/operator/balanced_interval_row_direction_warp_blocking_operator.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB.o  \
${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction.o \
${BUILDER_HOME}/operator/balanced_interval_row_direction_tblock_blocking_operator.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMTB_after_nnz_blocking_in_row_direction.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMTB_after_nnz_blocking_in_row_direction.o ${BUILDER_HOME}/operator/fixed_interval_nnz_direction_thread_blocking_operator.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMTB_after_fixed_blocking_in_nnz_direction.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMTB_after_fixed_blocking_in_nnz_direction.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW.o \
${BUILDER_HOME}/transform_step/get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB.o ${BUILDER_HOME}/transform_step/modify_col_indices_by_nnz_pad.o ${BUILDER_HOME}/transform_step/modify_row_indices_by_nnz_pad.o ${BUILDER_HOME}/transform_step/modify_vals_by_nnz_pad.o \
${BUILDER_HOME}/operator/fixed_interval_nnz_direction_warp_blocking_operator.o ${BUILDER_HOME}/operator/fixed_interval_nnz_direction_tblock_blocking_operator.o ${BUILDER_HOME}/kernel_token/basic_glue_code.o ${BUILDER_HOME}/code_generator.o ${BUILDER_HOME}/reduction_token/one_register_result_IO_of_reduction.o ${BUILDER_HOME}/reduction_token/total_BMT_result_reduce_to_one_register_token.o \
${BUILDER_HOME}/operator/fixed_interval_nnz_direction_warp_blocking_operator.o ${BUILDER_HOME}/operator/fixed_interval_nnz_direction_tblock_blocking_operator.o ${BUILDER_HOME}/operator/grid_block_operator.o ${BUILDER_HOME}/operator/thread_total_reduce_operator.o ${BUILDER_HOME}/kernel_token/arr_declaration_token.o ${BUILDER_HOME}/kernel_token/if_else_token.o \
${BUILDER_HOME}/reduction_token/glue_code_token.o ${BUILDER_HOME}/executor.o ${BUILDER_HOME}/transform_step/thread_bit_map.o ${BUILDER_HOME}/transform_step/parent_bit_map_of_thread.o ${BUILDER_HOME}/transform_step/segment_empty_row_indices.o ${BUILDER_HOME}/transform_step/segment_empty_flag.o ${BUILDER_HOME}/transform_step/segment_ptr.o ${BUILDER_HOME}/transform_step/get_begin_rows_after_merge_thread.o \
${BUILDER_HOME}/transform_step/get_begin_nzs_after_merge_thread.o ${BUILDER_HOME}/transform_step/get_begin_rows_relative_to_parent_after_merge_thread.o ${BUILDER_HOME}/transform_step/get_begin_nzs_relative_to_parent_after_merge_thread.o ${BUILDER_HOME}/transform_step/get_begin_BMTs_after_merge_thread.o ${BUILDER_HOME}/reduction_token/thread_bit_map_reduce_to_two_register_token.o \
${BUILDER_HOME}/reduction_token/total_warp_result_reduce_to_one_register_token.o ${BUILDER_HOME}/reduction_token/warp_bit_map_reduce_token.o ${BUILDER_HOME}/reduction_token/warp_segment_reduce_token.o ${BUILDER_HOME}/reduction_token/total_block_reduce_to_one_register_token.o ${BUILDER_HOME}/reduction_token/tblock_bit_map_reduce_token.o ${BUILDER_HOME}/reduction_token/tblock_segment_reduce_token.o \
${BUILDER_HOME}/transform_step/segment_offset.o ${BUILDER_HOME}/operator/thread_bit_map_operator.o ${BUILDER_HOME}/operator/warp_total_reduce_operator.o ${BUILDER_HOME}/operator/warp_bit_map_operator.o ${BUILDER_HOME}/operator/warp_segment_reduce_operator.o ${BUILDER_HOME}/operator/tblock_total_reduce_operator.o ${BUILDER_HOME}/operator/tblock_thread_bit_map_operator.o \
${BUILDER_HOME}/reduction_token/two_register_result_IO_of_reduction.o ${BUILDER_HOME}/operator/merge_path_tblock_operator.o ${BUILDER_HOME}/operator/merge_path_warp_operator.o ${BUILDER_HOME}/operator/merge_path_thread_operator.o ${BUILDER_HOME}/transform_step/get_begin_rows_of_level_after_merge_path.o ${BUILDER_HOME}/transform_step/get_begin_nzs_of_level_after_merge_path.o \
# ${BUILDER_HOME}/reduction_token/total_BMT_result_reduce_to_one_register_with_shared_sparse_token.o






token_test: ${src} ${BUILDER_HOME}/token_test.o
	${LD} -o $@ $^ ${LDFLAGS}


PHONY: clean
clean:
	rm -f *.o token_test ${src}