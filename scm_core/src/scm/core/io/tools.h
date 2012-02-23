
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_IO_TOOLS_H_INCLUDED
#define SCM_CORE_IO_TOOLS_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

__scm_export(core) bool
read_text_file(const std::string& in_file_path, std::string& out_file_string);

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_IO_TOOLS_H_INCLUDED
