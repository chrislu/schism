
#ifndef CONSOLE_OUTPUT_LISTENER_H_INCLUDED
#define CONSOLE_OUTPUT_LISTENER_H_INCLUDED

#include <scm/core/console/console_output.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace con {

class __scm_export(core) console_output_listener
{
public:
    console_output_listener();
    virtual ~console_output_listener();


    bool                                connect(console_out_stream& con);
    bool                                disconnect();

    void                                set_log_threshold(int               /*threshold*/);

protected:
    virtual void                        update(const std::string&           /*update_buffer*/,
                                               const console_out_stream&    /*stream_source*/) = 0;

    int                                 _log_threshold;

private:
    console_out_stream::connection_type _connection;

}; // class console_output_listener

} // namespace con
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // CONSOLE_OUTPUT_LISTENER_H_INCLUDED
