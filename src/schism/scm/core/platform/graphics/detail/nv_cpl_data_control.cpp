
#include "nv_cpl_data_control.h"

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS // #ifdef _WIN32
  #include <windows.h>
#else
#pragma warn (nv_cpl_data_control only working on windows platforms)
#endif

namespace
{
    static HINSTANCE    nv_cpl_lib;
} // namespace

using namespace scm::platform::detail;


bool nv_cpl_data_control::initialize_cpl_control()
{
#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS // #ifdef _WIN32
    if (nv_cpl_lib == 0) {
        nv_cpl_lib = LoadLibrary("nvcpl.dll");
        if (nv_cpl_lib == 0) {
            return (false);
        }
        else {
            _get_data_int = (nv_cpl_get_data_int_ptr)::GetProcAddress(nv_cpl_lib, "NvCplGetDataInt");
            _set_data_int = (nv_cpl_set_data_int_ptr)::GetProcAddress(nv_cpl_lib, "NvCplSetDataInt");

            if (   _get_data_int == 0
                || _set_data_int == 0) {

                close_cpl_control();
                return (false);
            }
        }
    }

    return (true);
#else
    return (false);
#endif
}

void nv_cpl_data_control::close_cpl_control()
{
#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS // #ifdef _WIN32
    if (nv_cpl_lib != 0) {
        FreeLibrary(nv_cpl_lib);

        nv_cpl_lib = 0;
        _get_data_int = 0;
        _set_data_int = 0;
    }
#endif
}

