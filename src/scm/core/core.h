
#ifndef SCM_CORE_H_INCLUDED
#define SCM_CORE_H_INCLUDED

#include <scm/core/int_types.h>
#include <scm/core/ptr_types.h>
#include <scm/core/platform/platform.h>

namespace scm {

bool __scm_export(core) initialize();
bool __scm_export(core) shutdown();

} // namespace scm

#endif // SCM_CORE_H_INCLUDED
