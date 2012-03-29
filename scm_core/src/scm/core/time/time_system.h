
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TIME_SYSTEM_H_INCLUDED
#define TIME_SYSTEM_H_INCLUDED

#include <scm/core/time/time_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace time {

__scm_export(core) ptime            local_time();
__scm_export(core) ptime            universal_time();

__scm_export(core) date             local_date();
__scm_export(core) date             universal_date();

} // namespace time
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TIME_SYSTEM_H_INCLUDED
