
#ifndef CORE_H_INCLUDED
#define CORE_H_INCLUDED

#include <scm_core/platform/platform.h>

#include <scm_core/int_types.h>
#include <scm_core/ptr_types.h>

#include <scm_core/root/global_system_access.h>


namespace scm
{
    namespace core
    {
        class system_root_interface;
        class console_interface;
        class script_system_interface;

        extern __scm_export global_system_access<system_root_interface>::type   root;
        extern __scm_export global_system_access<console_interface>::type       console;
        extern __scm_export global_system_access<script_system_interface>::type script_system;

    } // namespace core

} // namespace scm

#endif // CORE_H_INCLUDED
