
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CL_CORE_OPENCL_UTIL_H_INCLUDED
#define SCM_CL_CORE_OPENCL_UTIL_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cl {
namespace util {

std::string
__scm_export(cl_core) cl_error_string(int error_code);

} // namespace util
} // namespace cl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CL_CORE_OPENCL_UTIL_H_INCLUDED
