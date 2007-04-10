
#include <scm_core/platform/platform.h>

#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(push)               // preserve warning settings
#pragma warning(disable : 4996)     // disable warning C4996: 'std::_Copy_opt' was declared deprecated

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400


