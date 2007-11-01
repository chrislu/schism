
#ifndef ROOT_H_INCLUDED
#define ROOT_H_INCLUDED

#include <scm/core/root/root_system.h>
#include <scm/core/core_system_singleton.h>
#include <scm/core/platform/platform.h>

namespace scm {

extern __scm_export(core) core::core_system_singleton<core::root_system>::type  root;

} // namespace scm

#endif // ROOT_H_INCLUDED
