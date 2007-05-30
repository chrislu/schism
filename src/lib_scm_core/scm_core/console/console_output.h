
#ifndef CONSOLE_OUTPUT_H_INCLUDED
#define CONSOLE_OUTPUT_H_INCLUDED

#include <sstream>
#include <string>
#include <deque>

#include <scm_core/utilities/boost_warning_disable.h>

#include <boost/signal.hpp>
#include <boost/noncopyable.hpp>

#include <scm_core/utilities/boost_warning_enable.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace con {

class console_output_listener;

enum log_levels
{
    debug       = 100,
    warning     = 50,
    error       = 20,
    panic       = 0
};

struct log_level;

class __scm_export console_out_stream : boost::noncopyable
{
private:
    typedef boost::signal<void (const std::string&, const console_out_stream&)> stream_updated_signal_type;
    typedef boost::signals::connection                      connection_type;

public:
    console_out_stream();
    virtual ~console_out_stream();

    int                                 get_log_level() const { return (_log_level); }

private:
    std::stringstream                   _stream;
    int                                 _log_level;                           

    stream_updated_signal_type          _output_stream_updated_signal;

    connection_type                     connect_output_listener(stream_updated_signal_type::slot_function_type listener);
    bool                                disconnect_output_listener(connection_type& listener_connection);

    void                                emit_stream_updated_signal();

    friend class scm::con::console_output_listener;

    // pass through std::ostream operator << to _output_buffer
    template <class T>
    friend console_out_stream& operator << (console_out_stream& con, const T& rhs);
    friend console_out_stream& operator << (console_out_stream& con, std::ostream& (*_Pfn)(std::ostream&));
    friend console_out_stream& operator << (console_out_stream& con, const log_level& level);

}; // class console_out_stream

struct log_level
{
    log_level(int l) : _level(l) {}

    int get_level() const { return (_level); }
private:
    int _level;
}; // struct log_level

} // namespace con
} // namespace scm

#include "console_output.inl"

#include <scm_core/utilities/platform_warning_enable.h>

#endif // CONSOLE_OUTPUT_H_INCLUDED
