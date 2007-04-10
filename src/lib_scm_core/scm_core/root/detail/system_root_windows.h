
#ifndef SYSTEM_ROOT_WINDOWS_H_INCLUDED
#define SYSTEM_ROOT_WINDOWS_H_INCLUDED


#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

#include <scm_core/root/system_root.h>

namespace scm
{
    namespace core
    {
        namespace detail
        {
            class __scm_export system_root_windows : public system_root_interface
            {
            public:
                system_root_windows();
                virtual ~system_root_windows();
            protected:
            private:
            }; // class system_root_windows

        } // namespace detail
    } // namespace core
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // SYSTEM_ROOT_WINDOWS_H_INCLUDED
