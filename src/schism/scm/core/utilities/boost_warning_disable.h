
#include <scm/core/platform/platform.h>

#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(push)               // preserve warning settings
#pragma warning(disable : 4103)     // disable warning C4103: alignment changed after including header, may be due to missing #pragma pack(pop)
//#pragma warning(disable : 4561)     // disable warning C4561: __fastcall' incompatible with the '/clr' option: converting to '__stdcall'
//#pragma warning(disable : 4793)     // disable warning C4793: 'reason' : causes native code generation for function 'function'

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400



