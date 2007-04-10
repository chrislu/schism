
#include "system_root.h"

#include <cassert>

#include <scm_core/core.h>
#include <scm_core/console/console.h>
#include <scm_core/script/script_system.h>

#include <scm_core/exception/system_exception.h>

namespace scm {
    namespace core {

        // instantiation of global access instances
        global_system_access<system_root_interface>::type   root            = global_system_access<system_root_interface>::type();
        global_system_access<console_interface>::type       console         = global_system_access<console_interface>::type();
        global_system_access<script_system_interface>::type script_system   = global_system_access<script_system_interface>::type();

        // implementation of system_root_interface class
        system_root_interface::system_root_interface()
        {
            // check if system was allready initialized
            if (root.get_ptr() != 0) {
                throw scm::core::system_exception("system allready initialized (root instance != 0)!");
            }
        }

        system_root_interface::~system_root_interface()
        {
        }


        void system_root_interface::setup_global_access_references()
        {
            scm::core::root.set_instance(this);
            scm::core::console.set_instance(_console.get());
            scm::core::script_system.set_instance(_script_system.get());
        }


    } // namespace core
} // namespace scm
