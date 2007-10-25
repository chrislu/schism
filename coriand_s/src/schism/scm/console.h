
#ifndef CONSOLE_H_INCLUDED
#define CONSOLE_H_INCLUDED

#include <scm/core/console/console_system.h>
#include <scm/core/core_system_singleton.h>
#include <scm/core/platform/platform.h>

namespace scm {

extern __scm_export(core) core::core_system_singleton<con::console_system>::type  console;

} // namespace scm

#endif // CONSOLE_H_INCLUDED
