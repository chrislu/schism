
#ifndef CONSOLE_SYSTEM_H_INCLUDED
#define CONSOLE_SYSTEM_H_INCLUDED

#include <scm/core/utilities/boost_warning_disable.h>

#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/sys_interfaces.h>
#include <scm/core/console/console_output.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace con {

class scm::con::console_output_listener;

class __scm_export(core) console_system : public scm::core::system
{
public:
    typedef std::deque<std::string>                         input_history_container;

public:
    console_system(input_history_container::size_type hs = 50);
    virtual ~console_system();
    
    // core::system interface
    bool                                initialize();
    bool                                shutdown();

    // input handling/execution
    void                                add_input(const std::string&);
    bool                                process_input();

protected:
    console_out_stream                  _out_stream;
    std::stringstream                   _input_buffer;

    input_history_container             _input_history;
    input_history_container::size_type  _input_history_max_length;

    void                                add_input_history(const std::string&);

private:
    friend class scm::con::console_output_listener;
    // pass through std::ostream operator << to _out_stream
    template <class T>
    friend console_system& operator << (console_system& con, const T& rhs);
    friend console_system& operator << (console_system& con, std::ostream& (*_Pfn)(std::ostream&));
    friend console_system& operator << (console_system& con, const log_level& level);

}; // class console_system

} // namespace con
} // namespace scm

#include "console_system.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // CONSOLE_SYSTEM_H_INCLUDED
