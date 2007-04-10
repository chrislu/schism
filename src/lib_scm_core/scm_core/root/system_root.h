
#ifndef SYSTEM_ROOT_H_INCLUDED
#define SYSTEM_ROOT_H_INCLUDED

#include <boost/scoped_ptr.hpp>


#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

#include <scm_core/sys_interfaces.h>


namespace scm
{
    namespace core
    {
        // forward declarations
        class console_interface;
        class script_system_interface;

        class system_root_factory;

        typedef enum __scm_export {
            SCM_NULL_SYSTEM     = 0x00,
            SCM_CONSOLE,
            SCM_SCRIPT_SYSTEM
        } system_identifier_t;

        class __scm_export system_root_interface : public scm::core::system_refreshable
        {
        public:
            bool            initialize() {return (false);};
            bool            shutdown() {return (false);};
            bool            frame() {return (false);};

            //scm::core::system*const get_system(scm::core::system_identifier_t sys_id);


        protected:
            // prevent direct instantiation
            system_root_interface();
            virtual ~system_root_interface();


        protected:
            boost::scoped_ptr<console_interface>        _console;
            boost::scoped_ptr<script_system_interface>  _script_system;


        private:
            //typedef std::map<system_identifier_t, scm::core::system*const>  system_container_t;

            void                    setup_global_access_references();

            //void                    setup_system(system_identifier_t,
            //                                     scm::core::system*const);

            //system_container_t      _systems;


            friend class system_root_factory;
        }; // class system_root_interface 

    } // namespace core
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // SYSTEM_ROOT_H_INCLUDED
