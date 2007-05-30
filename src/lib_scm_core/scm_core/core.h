

#ifndef CORE_H_INCLUDED
#define CORE_H_INCLUDED

#include <scm_core/core/int_types.h>
#include <scm_core/core/ptr_types.h>
#include <scm_core/core/root_system.h>
#include <scm_core/platform/platform.h>

#include <scm_core/console.h>

namespace scm {

extern __scm_export core::core_system_singleton<core::root_system>::type  root;

bool __scm_export initialize();
bool __scm_export shutdown();

} // namespace scm

#endif // CORE_H_INCLUDED
