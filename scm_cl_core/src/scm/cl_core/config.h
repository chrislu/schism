
// $copyright$
// $license$

#ifndef SCM_CL_CORE_CONFIG_H_INCLUDED
#define SCM_CL_CORE_CONFIG_H_INCLUDED

#include <scm/core/platform/platform.h>

// gl debugging
#if SCM_DEBUG
#   define SCM_CL_DEBUG 0
#else
#   define SCM_CL_DEBUG 0
#endif

#define SCM_CL_CORE_OPENCL_ENABLE_PROFILING 1
#undef SCM_CL_CORE_OPENCL_ENABLE_PROFILING



#endif // SCM_CL_CORE_CONFIG_H_INCLUDED
