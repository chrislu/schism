
#ifndef ROOT_SYSTEM_H_INCLUDED
#define ROOT_SYSTEM_H_INCLUDED

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <scm_core/core/basic_system_interfaces.h>
#include <scm_core/core/ptr_types.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {

namespace con {
class console_system;
class console_output_listener;
class console_output_listener_stdout;
} // namespace con

namespace res {
class resource_manager_base;
} // namespace res

namespace time {
class time_system;
} // namespace time

namespace core {

class __scm_export root_system : public system
{
protected:
    typedef std::vector<std::pair<scm::core::system*, std::string> >    system_container;
    typedef std::map<std::string, system_container::size_type>          system_assoc_container;

    typedef std::map<std::size_t, res::resource_manager_base*const>     resource_manager_container;

public:
    root_system();
    virtual ~root_system();

    bool                            initialize();
    bool                            shutdown();

    std::string                     get_version_string() const;

    // subsystems
    core::system&                   get_subsystem(const std::string& /*sys_name*/);
    con::console_output_listener&   get_std_console_listener();

    // resource managers
    void                            register_resource_manager(res::resource_manager_base*const /*man*/,
                                                              std::size_t                      /*id*/);
    void                            unregister_resource_manager(std::size_t /*id*/);
    res::resource_manager_base&     get_resource_manager(std::size_t /*id*/);


protected:
    scoped_ptr<con::console_system>                     _console;
    scoped_ptr<con::console_output_listener_stdout>     _console_listener;
    scoped_ptr<time::time_system>                       _timing;

    void                            setup_global_system_singletons();
    void                            reset_global_system_singletons();

    // subsystems
    system_container                _subsystems;
    system_assoc_container          _subsystems_named;

    void                            register_subsystem(const std::string& /*sys_name*/,
                                                       scm::core::system* /*sys_ptr*/);
    void                            unregister_subsystem(const std::string& /*sys_name*/);
    void                            unregister_subsystems();

    bool                            initialize_subsystems();
    void                            shutdown_subsystems();

    // resource managers
    resource_manager_container      _resource_managers;
    void                            unregister_resource_managers();

private:

}; // class root_system

} // namespace core
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // ROOT_SYSTEM_H_INCLUDED
