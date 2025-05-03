// kernel的头文件，主要是从文件读入稀疏格式的相关索引
#include <cuda_fp16.h>

#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string.h>
#include <iostream>
#include <assert.h>

#include <sys/time.h>

#include <cctype>
#include <cstdio>
#include <cstring>

#include <math.h>
#include <algorithm>
#include <cassert>
#include <tuple>
#include <typeinfo>

using namespace std;


#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);

int mm_write_banner(FILE *f, MM_typecode matcode);
int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
int mm_write_mtx_array_size(FILE *f, int M, int N);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode) ((typecode)[0] == 'M')

#define mm_is_sparse(typecode) ((typecode)[1] == 'C')
#define mm_is_coordinate(typecode) ((typecode)[1] == 'C')
#define mm_is_dense(typecode) ((typecode)[1] == 'A')
#define mm_is_array(typecode) ((typecode)[1] == 'A')

#define mm_is_complex(typecode) ((typecode)[2] == 'C')
#define mm_is_real(typecode) ((typecode)[2] == 'R')
#define mm_is_pattern(typecode) ((typecode)[2] == 'P')
#define mm_is_integer(typecode) ((typecode)[2] == 'I')

#define mm_is_symmetric(typecode) ((typecode)[3] == 'S')
#define mm_is_general(typecode) ((typecode)[3] == 'G')
#define mm_is_skew(typecode) ((typecode)[3] == 'K')
#define mm_is_hermitian(typecode) ((typecode)[3] == 'H')

int mm_is_valid(MM_typecode matcode); /* too complex for a macro */

/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode) ((*typecode)[0] = 'M')
#define mm_set_coordinate(typecode) ((*typecode)[1] = 'C')
#define mm_set_array(typecode) ((*typecode)[1] = 'A')
#define mm_set_dense(typecode) mm_set_array(typecode)
#define mm_set_sparse(typecode) mm_set_coordinate(typecode)

#define mm_set_complex(typecode) ((*typecode)[2] = 'C')
#define mm_set_real(typecode) ((*typecode)[2] = 'R')
#define mm_set_pattern(typecode) ((*typecode)[2] = 'P')
#define mm_set_integer(typecode) ((*typecode)[2] = 'I')

#define mm_set_symmetric(typecode) ((*typecode)[3] = 'S')
#define mm_set_general(typecode) ((*typecode)[3] = 'G')
#define mm_set_skew(typecode) ((*typecode)[3] = 'K')
#define mm_set_hermitian(typecode) ((*typecode)[3] = 'H')

#define mm_clear_typecode(typecode)                                            \
  ((*typecode)[0] = (*typecode)[1] = (*typecode)[2] = ' ', (*typecode)[3] = 'G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

/********************* Matrix Market error codes ***************************/

#define MM_COULD_NOT_READ_FILE 11
#define MM_PREMATURE_EOF 12
#define MM_NOT_MTX 13
#define MM_NO_HEADER 14
#define MM_UNSUPPORTED_TYPE 15
#define MM_LINE_TOO_LONG 16
#define MM_COULD_NOT_WRITE_FILE 17

/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

            ojbect    sparse/     data        storage
                  dense       type        scheme

   string position:  [0]        [1]     [2]         [3]

   Matrix typecode:  M(atrix)  C(oord)    R(eal)    G(eneral)
                    A(array)  C(omplex)   H(ermitian)
                      P(attern)   S(ymmetric)
                        I(nteger) K(kew)

 ***********************************************************************/

#define MM_MTX_STR "matrix"
#define MM_ARRAY_STR "array"
#define MM_DENSE_STR "array"
#define MM_COORDINATE_STR "coordinate"
#define MM_SPARSE_STR "coordinate"
#define MM_COMPLEX_STR "complex"
#define MM_REAL_STR "real"
#define MM_INT_STR "integer"
#define MM_GENERAL_STR "general"
#define MM_SYMM_STR "symmetric"
#define MM_HERM_STR "hermitian"
#define MM_SKEW_STR "skew-symmetric"
#define MM_PATTERN_STR "pattern"


/************************************************************************
 
type define for vector load and store

 *************************************************************************/

struct short8
{
  short4 x;
  short4 y;
};

struct char8
{
  char4 x;
  char4 y;
};

struct half4
{
  half2 x;
  half2 y;
};

struct half8
{
  half4 x;
  half4 y;
};





/*  high level routines */

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
                     double val[], MM_typecode matcode);
int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
                         double val[], MM_typecode matcode);
int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *img,
                          MM_typecode matcode);

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                               double **val_, int **I_, int **J_);

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                               double **val_, int **I_, int **J_) {
  FILE *f;
  MM_typecode matcode;
  int M, N, nz;
  int i;
  double *val;
  int *I, *J;

  if ((f = fopen(fname, "r")) == NULL)
    return -1;

  if (mm_read_banner(f, &matcode) != 0) {
    printf("mm_read_unsymetric: Could not process Matrix Market banner ");
    printf(" in file [%s]\n", fname);
    return -1;
  }

  if (!(mm_is_real(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode))) {
    fprintf(stderr, "Sorry, this application does not support ");
    fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    return -1;
  }

  /* find out size of sparse matrix: M, N, nz .... */

  if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
    fprintf(stderr,
            "read_unsymmetric_sparse(): could not parse matrix size.\n");
    return -1;
  }

  *M_ = M;
  *N_ = N;
  *nz_ = nz;

  /* reseve memory for matrices */

  I = (int *)malloc(nz * sizeof(int));
  J = (int *)malloc(nz * sizeof(int));
  val = (double *)malloc(nz * sizeof(double));

  *val_ = val;
  *I_ = I;
  *J_ = J;

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (i = 0; i < nz; i++) {
    int u = fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
    I[i]--; /* adjust from 1-based to 0-based */
    J[i]--;
  }
  fclose(f);

  return 0;
}

int mm_is_valid(MM_typecode matcode) {
  if (!mm_is_matrix(matcode))
    return 0;
  if (mm_is_dense(matcode) && mm_is_pattern(matcode))
    return 0;
  if (mm_is_real(matcode) && mm_is_hermitian(matcode))
    return 0;
  if (mm_is_pattern(matcode) &&
      (mm_is_hermitian(matcode) || mm_is_skew(matcode)))
    return 0;
  return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode) {
  char line[MM_MAX_LINE_LENGTH];
  char banner[MM_MAX_TOKEN_LENGTH];
  char mtx[MM_MAX_TOKEN_LENGTH];
  char crd[MM_MAX_TOKEN_LENGTH];
  char data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  char *p;

  mm_clear_typecode(matcode);

  if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
    return MM_PREMATURE_EOF;

  if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
             storage_scheme) != 5)
    return MM_PREMATURE_EOF;

  for (p = mtx; *p != '\0'; *p = tolower(*p), p++)
    ; /* convert to lower case */
  for (p = crd; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = data_type; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++)
    ;

  /* check for banner */
  if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
    return MM_NO_HEADER;

  /* first field should be "mtx" */
  if (strcmp(mtx, MM_MTX_STR) != 0)
    return MM_UNSUPPORTED_TYPE;
  mm_set_matrix(matcode);

  /* second field describes whether this is a sparse matrix (in coordinate
          storgae) or a dense array */

  if (strcmp(crd, MM_SPARSE_STR) == 0)
    mm_set_sparse(matcode);
  else if (strcmp(crd, MM_DENSE_STR) == 0)
    mm_set_dense(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  /* third field */

  if (strcmp(data_type, MM_REAL_STR) == 0)
    mm_set_real(matcode);
  else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
    mm_set_complex(matcode);
  else if (strcmp(data_type, MM_PATTERN_STR) == 0)
    mm_set_pattern(matcode);
  else if (strcmp(data_type, MM_INT_STR) == 0)
    mm_set_integer(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  /* fourth field */

  if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
    mm_set_general(matcode);
  else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
    mm_set_symmetric(matcode);
  else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
    mm_set_hermitian(matcode);
  else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
    mm_set_skew(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz) {
  if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
    return MM_COULD_NOT_WRITE_FILE;
  else
    return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz) {
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;

  /* set return null parameter values, in case we exit with errors */
  *M = *N = *nz = 0;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  } while (line[0] == '%');

  /* line[] is either blank or has M,N, nz */
  if (sscanf(line, "%d %d %d", M, N, nz) == 3)
    return 0;

  else
    do {
      num_items_read = fscanf(f, "%d %d %d", M, N, nz);
      if (num_items_read == EOF)
        return MM_PREMATURE_EOF;
    } while (num_items_read != 3);

  return 0;
}

int mm_read_mtx_array_size(FILE *f, int *M, int *N) {
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;
  /* set return null parameter values, in case we exit with errors */
  *M = *N = 0;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  } while (line[0] == '%');

  /* line[] is either blank or has M,N, nz */
  if (sscanf(line, "%d %d", M, N) == 2)
    return 0;

  else /* we have a blank line */
    do {
      num_items_read = fscanf(f, "%d %d", M, N);
      if (num_items_read == EOF)
        return MM_PREMATURE_EOF;
    } while (num_items_read != 2);

  return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N) {
  if (fprintf(f, "%d %d\n", M, N) != 2)
    return MM_COULD_NOT_WRITE_FILE;
  else
    return 0;
}

/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
                         double val[], MM_typecode matcode) {
  int i;
  if (mm_is_complex(matcode)) {
    for (i = 0; i < nz; i++)
      if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2 * i],
                 &val[2 * i + 1]) != 4)
        return MM_PREMATURE_EOF;
  } else if (mm_is_real(matcode)) {
    for (i = 0; i < nz; i++) {
      if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3)
        return MM_PREMATURE_EOF;
    }
  }

  else if (mm_is_pattern(matcode)) {
    for (i = 0; i < nz; i++)
      if (fscanf(f, "%d %d", &I[i], &J[i]) != 2)
        return MM_PREMATURE_EOF;
  } else
    return MM_UNSUPPORTED_TYPE;

  return 0;
}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *imag,
                          MM_typecode matcode) {
  if (mm_is_complex(matcode)) {
    if (fscanf(f, "%d %d %lg %lg", I, J, real, imag) != 4)
      return MM_PREMATURE_EOF;
  } else if (mm_is_real(matcode)) {
    if (fscanf(f, "%d %d %lg\n", I, J, real) != 3)
      return MM_PREMATURE_EOF;

  }

  else if (mm_is_pattern(matcode)) {
    if (fscanf(f, "%d %d", I, J) != 2)
      return MM_PREMATURE_EOF;
  } else
    return MM_UNSUPPORTED_TYPE;

  return 0;
}

/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J,
                    double **val, MM_typecode *matcode) {
  int ret_code;
  FILE *f;

  if (strcmp(fname, "stdin") == 0)
    f = stdin;
  else if ((f = fopen(fname, "r")) == NULL)
    return MM_COULD_NOT_READ_FILE;

  if ((ret_code = mm_read_banner(f, matcode)) != 0)
    return ret_code;

  if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) &&
        mm_is_matrix(*matcode)))
    return MM_UNSUPPORTED_TYPE;

  if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
    return ret_code;

  *I = (int *)malloc(*nz * sizeof(int));
  *J = (int *)malloc(*nz * sizeof(int));
  *val = NULL;

  if (mm_is_complex(*matcode)) {
    *val = (double *)malloc(*nz * 2 * sizeof(double));
    ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, *matcode);
    if (ret_code != 0)
      return ret_code;
  } else if (mm_is_real(*matcode)) {
    *val = (double *)malloc(*nz * sizeof(double));
    ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, *matcode);
    if (ret_code != 0)
      return ret_code;
  }

  else if (mm_is_pattern(*matcode)) {
    ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, *matcode);
    if (ret_code != 0)
      return ret_code;
  }

  if (f != stdin)
    fclose(f);
  return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode) {
  char *str = mm_typecode_to_str(matcode);
  int ret_code;

  ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
  free(str);
  if (ret_code != 2)
    return MM_COULD_NOT_WRITE_FILE;
  else
    return 0;
}

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
                     double val[], MM_typecode matcode) {
  FILE *f;
  int i;

  if (strcmp(fname, "stdout") == 0)
    f = stdout;
  else if ((f = fopen(fname, "w")) == NULL)
    return MM_COULD_NOT_WRITE_FILE;

  /* print banner followed by typecode */
  fprintf(f, "%s ", MatrixMarketBanner);
  fprintf(f, "%s\n", mm_typecode_to_str(matcode));

  /* print matrix sizes and nonzeros */
  fprintf(f, "%d %d %d\n", M, N, nz);

  /* print values */
  if (mm_is_pattern(matcode))
    for (i = 0; i < nz; i++)
      fprintf(f, "%d %d\n", I[i], J[i]);
  else if (mm_is_real(matcode))
    for (i = 0; i < nz; i++)
      fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
  else if (mm_is_complex(matcode))
    for (i = 0; i < nz; i++)
      fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2 * i],
              val[2 * i + 1]);
  else {
    if (f != stdout)
      fclose(f);
    return MM_UNSUPPORTED_TYPE;
  }

  if (f != stdout)
    fclose(f);

  return 0;
}

/**
 *  Create a new copy of a string s.  mm_strdup() is a common routine, but
 *  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
 *
 */
char *mm_strdup(const char *s) {
  int len = strlen(s);
  char *s2 = (char *)malloc((len + 1) * sizeof(char));
  return strcpy(s2, s);
}

char *mm_typecode_to_str(MM_typecode matcode) {
  char buffer[MM_MAX_LINE_LENGTH];
  char const *types[4];
  char *mm_strdup(const char *);

  /* check for MTX type */
  if (mm_is_matrix(matcode))
    types[0] = MM_MTX_STR;

  /* check for CRD or ARR matrix */
  if (mm_is_sparse(matcode))
    types[1] = MM_SPARSE_STR;
  else if (mm_is_dense(matcode))
    types[1] = MM_DENSE_STR;
  else
    return NULL;

  /* check for element data type */
  if (mm_is_real(matcode))
    types[2] = MM_REAL_STR;
  else if (mm_is_complex(matcode))
    types[2] = MM_COMPLEX_STR;
  else if (mm_is_pattern(matcode))
    types[2] = MM_PATTERN_STR;
  else if (mm_is_integer(matcode))
    types[2] = MM_INT_STR;
  else
    return NULL;

  /* check for symmetry type */
  if (mm_is_general(matcode))
    types[3] = MM_GENERAL_STR;
  else if (mm_is_symmetric(matcode))
    types[3] = MM_SYMM_STR;
  else if (mm_is_hermitian(matcode))
    types[3] = MM_HERM_STR;
  else if (mm_is_skew(matcode))
    types[3] = MM_SKEW_STR;
  else
    return NULL;

  sprintf(buffer, "%s %s %s %s", types[0], types[1], types[2], types[3]);
  return mm_strdup(buffer);
}


enum data_type
{
    CHAR,
    UNSIGNED_CHAR,
    SHORT,
    UNSIGNED_SHORT,
    INT,
    UNSIGNED_INT,
    LONG,
    UNSIGNED_LONG,
    LONG_LONG,
    UNSIGNED_LONG_LONG,
    FLOAT,
    DOUBLE,
    BOOL
};

void *malloc_arr(unsigned long length, data_type type_of_arr)
{
    assert(type_of_arr == UNSIGNED_CHAR || type_of_arr == UNSIGNED_INT ||
           type_of_arr == UNSIGNED_SHORT || type_of_arr == UNSIGNED_LONG ||
           type_of_arr == DOUBLE || type_of_arr == FLOAT);

    assert(length > 0);

    if (type_of_arr == UNSIGNED_CHAR)
    {
        return new unsigned char[length];
    }
    else if (type_of_arr == UNSIGNED_SHORT)
    {
        return new unsigned short[length];
    }
    else if (type_of_arr == UNSIGNED_INT)
    {
        unsigned int *return_ptr = new unsigned int[length];
        return (void *)return_ptr;
    }
    else if (type_of_arr == DOUBLE)
    {
        return new double[length];
    }
    else if (type_of_arr == FLOAT)
    {
        return new float[length];
    }
    else
    {
        return new unsigned long[length];
    }
}

void write_double_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, double write_val)
{
    assert(type == DOUBLE || type == FLOAT);

    if (type == DOUBLE)
    {
        double *input_arr = (double *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == FLOAT)
    {
        float *input_arr = (float *)arr;
        input_arr[write_pos] = write_val;
    }
}

void write_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, unsigned long write_val)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR);
    if (type == UNSIGNED_LONG)
    {
        unsigned long *input_arr = (unsigned long *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *input_arr = (unsigned int *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *input_arr = (unsigned short *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *input_arr = (unsigned char *)arr;
        input_arr[write_pos] = write_val;
    }
}

void *read_arr_from_file_with_data_type(unsigned long length, data_type arr_data_type, string file_name)
{
    assert(length > 0);
    assert(arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR ||
           arr_data_type == BOOL || arr_data_type == FLOAT || arr_data_type == DOUBLE);

    void *arr_need_to_return = malloc_arr(length, arr_data_type);

    unsigned long cur_insert_index = 0;

    if (arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR || arr_data_type == BOOL)
    {
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

        if (infile.is_open())
        {
            while (infile.good() && !infile.eof())
            {
                string line_str;
                memset(buf, 0, 1024);
                infile.getline(buf, 1024);
                line_str = buf;

                if (isspace(line_str[0]) || line_str.empty())
                {
                    continue;
                }

                unsigned long arr_val = atol(line_str.c_str());

                assert(cur_insert_index < length);
                write_to_array_with_data_type(arr_need_to_return, arr_data_type, cur_insert_index, arr_val);

                cur_insert_index++;
            }
        }
        
        assert(cur_insert_index == length);
        infile.close();
        return arr_need_to_return;
    }
    else if (arr_data_type == DOUBLE || arr_data_type == FLOAT)
    {
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

        while (infile.good() && !infile.eof())
        {
            string line_str;
            memset(buf, 0, 1024);
            infile.getline(buf, 1024);
            line_str = buf;

            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            double arr_val = stod(line_str.c_str());

            assert(cur_insert_index < length);
            write_double_to_array_with_data_type(arr_need_to_return, arr_data_type, cur_insert_index, arr_val);

            cur_insert_index++;
        }
        
        assert(cur_insert_index == length);
        infile.close();
        return arr_need_to_return;
    }

    return arr_need_to_return;
}

unsigned long read_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR || type == BOOL);

    if (type == UNSIGNED_LONG)
    {
        unsigned long *output_arr = (unsigned long *)arr;
        return (unsigned long)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *output_arr = (unsigned int *)arr;
        return (unsigned long)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *output_arr = (unsigned short *)arr;
        return (unsigned short)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *output_arr = (unsigned char *)arr;
        return (unsigned char)(output_arr[read_pos]);
    }

    if (type == BOOL)
    {
        bool *output_arr = (bool *)arr;
        return (bool)(output_arr[read_pos]);
    }

    cout << "error" << endl;
    exit(-1);
    return 0;
}

double read_double_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
{
    assert(type == DOUBLE || type == FLOAT);

    if (type == DOUBLE)
    {
        double *output_arr = (double *)arr;
        return (double)(output_arr[read_pos]);
    }

    if (type == FLOAT)
    {
        float *output_arr = (float *)arr;
        return (double)(output_arr[read_pos]);
    }

    return 0;
}


template <typename DType> void fill_zero(DType array[], int size) {
  memset(array, 0x0, sizeof(array[0]) * size);
}

void fill_one(float array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = 1.0;
  }
}

void fill_one_half(half array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = __float2half(1.0);
  }
}


template <typename Index, typename DType>
void spmm_reference_host(
    int M, 
    int N, 
    int K, 
    const Index *csr_indptr, const int *csr_indices,
    const DType *csr_values,
    const DType *B,         
    DType *C_ref)            
{
  fill_zero(C_ref, M * N);
  for (int64_t i = 0; i < M; i++) {
    Index begin = csr_indptr[i];
    Index end = csr_indptr[i + 1];
    for (Index p = begin; p < end; p++) {
      int k = csr_indices[p];
      DType val = csr_values[p];
      for (int64_t j = 0; j < N; j++) {
        C_ref[i * N + j] += val * B[k * N + j];
      }
    }
  }
}


template <typename DType>
bool check_result(int M, int N, DType *C, DType *C_ref, bool flag = false) {
  bool passed = true;
  int count = 0;
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      DType c = C[i * N + j];
      DType c_ref = C_ref[i * N + j];
      if ((c - c_ref != c_ref - c)) {
        if(count < 10)
        {
          if (flag == true)
          {
            printf(
                "Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
                i, j, __half2float(c), __half2float(c_ref));
          }
          else
          {
            printf(
                "Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
                i, j, c, c_ref);
          }
        }

        count += 1;
        passed = false;
      }
    }
  }
  printf("wrong number:%d\n", count);

  if (passed == true)
  {
    printf("correct\n");
  }
  return passed;
}

void read_mtx_file(const char *filename, int &nrow, int &ncol, int &nnz,
                   vector<int> &csr_indptr_buffer,
                   vector<int> &csr_indices_buffer) {
  FILE *f;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("File %s not found", filename);
    exit(EXIT_FAILURE);
  }
  cout << filename << endl;
  MM_typecode matcode;
  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  // printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  /// read tuples

  vector<tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_integer(matcode) || mm_is_real(matcode)) {
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      coords.push_back(make_tuple(row_id - 1, col_id - 1));
    }
  }

  /// make symmetric

  if (mm_is_symmetric(matcode)) {
    vector<tuple<int, int>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = get<0>(*iter);
      int j = get<1>(*iter);

      new_coords.push_back(make_tuple(i, j));
      new_coords.push_back(make_tuple(j, i));
    }
    sort(new_coords.begin(), new_coords.end());
    coords.clear();
    for (auto iter = new_coords.begin(); iter != new_coords.end(); iter++) {
      if ((iter + 1) == new_coords.end() || (*iter != *(iter + 1))) {
        coords.push_back(*iter);
      }
    }
  } else {
    sort(coords.begin(), coords.end());
  }

  /// generate csr from coo

  csr_indptr_buffer.clear();
  csr_indices_buffer.clear();

  int curr_pos = 0;
  csr_indptr_buffer.push_back(0);
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && (get<0>(coords[curr_pos]) == row)) {
      csr_indices_buffer.push_back(get<1>(coords[curr_pos]));
      curr_pos++;
    }
    // assert((get<0>(coords[curr_pos]) > row || curr_pos == nnz));
    csr_indptr_buffer.push_back(curr_pos);
  }

  nnz = csr_indices_buffer.size();
}


template <class To, class From>
__device__ __forceinline__ To BitCast(const From& src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}


__device__ __forceinline__ half Load(const half* address) {
  half x = __ldg(reinterpret_cast<const half*>(address));
  return (half)(x);
}


__device__ __forceinline__ half2 Load(const half2* address) {
  half2 x = __ldg(reinterpret_cast<const half2*>(address));
  return (half2)(x);
}


__device__ __forceinline__ half4 Load(const half4* address) {
  float2 x = __ldg(reinterpret_cast<const float2*>(address));
  return BitCast<half4>(x);
}

__device__ __forceinline__ half8 Load(const half8* address) {
  float4 x = __ldg(reinterpret_cast<const float4*>(address));
  return BitCast<half8>(x);
}


__device__ __forceinline__ float Load(const float* address) {
  float x = __ldg(reinterpret_cast<const float*>(address));
  return (float)(x);
}


__device__ __forceinline__ float2 Load(const float2* address) {
  float2 x = __ldg(reinterpret_cast<const float2*>(address));
  return (float2)(x);
}


__device__ __forceinline__ float4 Load(const float4* address) {
  float4 x = __ldg(reinterpret_cast<const float4*>(address));
  return (float4)(x);
}

__device__ __forceinline__ double Load(const double* address) {
  double x = __ldg(reinterpret_cast<const double*>(address));
  return (double)(x);
}

__device__ __forceinline__ double2 Load(const double2* address) {
  float4 x = __ldg(reinterpret_cast<const float4*>(address));
  return BitCast<double2>(x);
}

__device__ __forceinline__ char Load(const char* address) {
  char x = __ldg(reinterpret_cast<const char*>(address));
  return (char)(x);
}

__device__ __forceinline__ char Load(const unsigned char* address) {
  char x = __ldg(reinterpret_cast<const unsigned char*>(address));
  return (unsigned char)(x);
}

__device__ __forceinline__ char2 Load(const char2* address) {
  short x = __ldg(reinterpret_cast<const short*>(address));
  return BitCast<char2>(x);
}

__device__ __forceinline__ char4 Load(const char4* address) {
  int x = __ldg(reinterpret_cast<const int*>(address));
  return BitCast<char4>(x);
}

__device__ __forceinline__ char8 Load(const char8* address) {
  int2 x = __ldg(reinterpret_cast<const int2*>(address));
  return BitCast<char8>(x);
}

__device__ __forceinline__ short Load(const short* address) {
  short x = __ldg(reinterpret_cast<const short*>(address));
  return (short)(x);
}

__device__ __forceinline__ short Load(const unsigned short* address) {
  short x = __ldg(reinterpret_cast<const unsigned short*>(address));
  return (unsigned short)(x);
}

__device__ __forceinline__ short2 Load(const short2* address) {
  int x = __ldg(reinterpret_cast<const int*>(address));
  return BitCast<short2>(x);
}

__device__ __forceinline__ short4 Load(const short4* address) {
  int2 x = __ldg(reinterpret_cast<const int2*>(address));
  return BitCast<short4>(x);
}

__device__ __forceinline__ short8 Load(const short8* address) {
  int4 x = __ldg(reinterpret_cast<const int4*>(address));
  return BitCast<short8>(x);
}

__device__ __forceinline__ int Load(const int* address) {
  int x = __ldg(reinterpret_cast<const int*>(address));
  return (int)(x);
}

__device__ __forceinline__ int Load(const unsigned int* address) {
  int x = __ldg(reinterpret_cast<const unsigned int*>(address));
  return (unsigned int)(x);
}


__device__ __forceinline__ int2 Load(const int2* address) {
  int2 x = __ldg(reinterpret_cast<const int2*>(address));
  return BitCast<int2>(x);
}

__device__ __forceinline__ int4 Load(const int4* address) {
  int4 x = __ldg(reinterpret_cast<const int4*>(address));
  return BitCast<int4>(x);
}

__device__ __forceinline__ long Load(const long* address) {
  long x = __ldg(reinterpret_cast<const long*>(address));
  return (long)(x);
}

__device__ __forceinline__ long Load(const unsigned long* address) {
  long x = __ldg(reinterpret_cast<const unsigned long*>(address));
  return (unsigned long)(x);
}

__device__ __forceinline__ long2 Load(const long2* address) {
  int4 x = __ldg(reinterpret_cast<const int4*>(address));
  return BitCast<long2>(x);
}
