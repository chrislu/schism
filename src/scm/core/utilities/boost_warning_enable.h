
#include <scm/core/platform/platform.h>

#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(pop)                // restore warnings to previous state

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400
