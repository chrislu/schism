
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

namespace time {
class time_system;
} // namespace time

namespace core {

class __scm_export root_system : public system
{
protected:
    typedef std::vector<std::pair<scm::core::system*, std::string> >    system_container;
    typedef std::map<std::string, system_container::size_type>          system_assoc_container;

public:
    root_system();
    virtual ~root_system();

    bool                            initialize();
    bool                            shutdown();

    std::string                     get_version_string() const;

    core::system&                   get_subsystem(const std::string& /*sys_name*/);
    con::console_output_listener&   get_std_console_listener();

protected:
    scoped_ptr<con::console_system>                     _console;
    scoped_ptr<con::console_output_listener_stdout>     _console_listener;
    scoped_ptr<time::time_system>                       _timing;

    void                    setup_global_system_singletons();
    void                    reset_global_system_singletons();

    system_container        _subsystems;
    system_assoc_container  _subsystems_named;

    void                    register_subsystem(const std::string& /*sys_name*/,
                                               scm::core::system* /*sys_ptr*/);
    void                    unregister_subsystem(const std::string& /*sys_name*/);
    void                    unregister_subsystems();

    bool                    initialize_subsystems();
    void                    shutdown_subsystems();


private:

}; // class root_system

} // namespace core
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // ROOT_SYSTEM_H_INCLUDED
