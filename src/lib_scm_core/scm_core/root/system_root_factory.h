
#ifndef SYSTEM_ROOT_FACTORY_H_INCLUDED
#define SYSTEM_ROOT_FACTORY_H_INCLUDED

#include <scm_core/platform/platform.h>

namespace scm
{
    namespace core
    {
        class system_root_interface;

        class __scm_export system_root_factory
        {
        public:
            static system_root_interface*   create_system_root();

        }; // class system_root_factory

    } // namespace core
} // namespace scm

#endif // SYSTEM_ROOT_FACTORY_H_INCLUDED
