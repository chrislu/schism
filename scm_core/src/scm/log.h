
#ifndef SCM_CORE_LOG_H_INCLUDED
#define SCM_CORE_LOG_H_INCLUDED

#include <string>

#include <scm/core/log/log_core.h>
#include <scm/core/log/logger.h>
#include <scm/core/log/out_stream.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {

typedef logging::logger log;
typedef logging::level  log_level;

__scm_export(core) log&                 logger(const std::string& name);

__scm_export(core) logging::out_stream  out();
__scm_export(core) logging::out_stream  err();

} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_H_INCLUDED
