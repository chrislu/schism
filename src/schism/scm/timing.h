
#ifndef TIMING_H_INCLUDED
#define TIMING_H_INCLUDED

#include <scm/core/time/time_system.h>
#include <scm/core/core_system_singleton.h>
#include <scm/core/platform/platform.h>

namespace scm {

extern __scm_export(core) core::core_system_singleton<time::time_system>::type  timing;

} // namespace scm

#endif // TIMING_H_INCLUDED
