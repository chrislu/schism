
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_H_INCLUDED
#define SCM_CORE_LOG_H_INCLUDED

#include <string>

#include <scm/core/log/core.h>
#include <scm/core/log/logger.h>
#include <scm/core/log/out_stream.h>
#include <scm/core/log/out_stream_manip.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {

//typedef log::logger log;
//typedef log::level  log_level;

__scm_export(core) log::logger&     logger(const std::string& name);

__scm_export(core) log::out_stream  out();
__scm_export(core) log::out_stream  err();

} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_H_INCLUDED
