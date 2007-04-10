
#include "system_root_factory.h"

#include <cassert>

#include <scm_core/root/system_root.h>

#if     SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    #include <scm_core/root/detail/system_root_windows.h>
#elif   SCM_PLATFORM == SCM_PLATFORM_LINUX
    #error "unsupported platform"
#elif   SCM_PLATFORM == SCM_PLATFORM_APPLE
    #error "unsupported platform"
#endif

using namespace scm::core;

system_root_interface* system_root_factory::create_system_root()
{
    system_root_interface* tmp_ret = 0;
    
#if     SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    tmp_ret = new scm::core::detail::system_root_windows();
#elif   SCM_PLATFORM == SCM_PLATFORM_LINUX
    #error "unsupported platform"
#elif   SCM_PLATFORM == SCM_PLATFORM_APPLE
    #error "unsupported platform"
#endif

    assert(tmp_ret != 0);

    tmp_ret->setup_global_access_references();
    
    return (tmp_ret);
}
